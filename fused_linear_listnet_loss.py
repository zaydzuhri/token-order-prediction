# -*- coding: utf-8 -*-

# Code adapted from
# https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py

from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle

from fla.ops.utils import logsumexp_fwd
from fla.ops.utils.op import exp
from fla.utils import input_guard

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576
# https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2

@triton.jit
def listnet_kernel(
    logits,
    targets,  # Now full target distributions
    lse_logits,
    lse_targets,
    loss,
    total,
    ignore_index,
    logit_scale: tl.constexpr,
    reduction: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr
):
    i_n = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    
    # Pointers to current token's data
    logits_ptr = logits + i_n * V
    targets_ptr = targets + i_n * V
    loss_ptr = loss + i_n
    
    # Compute prediction softmax
    b_lse_logits = tl.load(lse_logits + i_n)
    b_lse_targets = tl.load(lse_targets + i_n)
    b_loss = 0.0
    
    # Compute gradient: softmax(pred) - softmax(target)
    for iv in range(0, NV):
        o_v = iv * BV + tl.arange(0, BV)
        mask = o_v < V
        
        # Load target and compute softmax
        t_val = tl.load(targets_ptr + o_v, mask=mask, other=0.0)
        p_target = tl.exp(t_val - b_lse_targets)
        
        # Load logits and compute softmax
        l_val = tl.load(logits_ptr + o_v, mask=mask, other=0.0) * logit_scale
        l_val_minus_lse = l_val - b_lse_logits
        p_pred = tl.exp(l_val_minus_lse)
        
        # Gradient calculation
        grad_val = p_pred - p_target
        if reduction == "mean":
            grad_val = grad_val / total
        grad_val = tl.where(b_lse_targets == float('inf'), 0.0, grad_val)
        tl.store(logits_ptr + o_v, grad_val, mask=mask)
        
        # ListNet loss
        # instead of: b_loss -= tl.sum(p_target * tl.log(p_pred), axis=0)
        b_loss -= tl.sum(p_target * l_val_minus_lse, axis=0)
    
    tl.store(loss_ptr, b_loss)

@triton.jit
def elementwise_mul_kernel(
    x,
    g,
    N: tl.constexpr,
    B: tl.constexpr
):
    """
    This function multiplies each element of the tensor pointed by x with the value pointed by g.
    The multiplication is performed in-place on the tensor pointed by x.

    Parameters:
    x:
        Pointer to the input tensor.
    g:
        Pointer to the gradient output value.
    N (int):
        The number of columns in the input tensor.
    B (int):
        The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    i_x = tl.program_id(0).to(tl.int64)
    o_x = i_x * B + tl.arange(0, B)

    # Load the gradient output value
    b_g = tl.load(g)
    b_x = tl.load(x + o_x, mask=o_x < N)
    tl.store(x + o_x, b_x * b_g, mask=o_x < N)


def fused_linear_listnet_forward(
    x: torch.Tensor,
    targets: torch.Tensor,  # Float tensor [N, V]
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    logit_scale: float = 1.0,
    num_chunks: int = 8,
    reduction: str = "mean"
):
    N, H, V = *x.shape, weight.shape[0]
    BV = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    NC = min(num_chunks, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)

    # Initialize outputs
    dx = torch.zeros_like(x)
    dw = torch.zeros_like(weight, dtype=torch.float) if weight is not None else None
    db = torch.zeros_like(bias, dtype=torch.float) if bias is not None else None
    loss = torch.zeros(N, device=x.device, dtype=torch.float)
    total = N  # All tokens considered

    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)
        c_x = x[start:end]
        c_logits = F.linear(c_x, weight, bias)
        c_targets = targets[start:end]
        c_lse_logits = logsumexp_fwd(c_logits, scale=logit_scale, dtype=torch.float)
        c_lse_targets = logsumexp_fwd(c_targets, dtype=torch.float).nan_to_num(nan=float("inf"))
        c_loss = loss[start:end]

        # Call ListNet kernel
        listnet_kernel[(c_logits.shape[0],)](
            logits=c_logits,
            targets=c_targets,  # Full target distributions
            lse_logits=c_lse_logits,
            lse_targets=c_lse_targets,
            loss=c_loss,
            total=total,
            ignore_index=ignore_index,
            logit_scale=logit_scale,
            reduction=reduction,
            V=V,
            BV=BV,
            num_warps=32
        )

        # Backward through linear layer
        dx[start:end] = torch.mm(c_logits, weight)
        if weight is not None:
            dw += c_logits.t() @ c_x
        if bias is not None:
            db += c_logits.sum(0)

    loss = loss.sum()
    if reduction == "mean":
        loss = loss / total
        
    return loss, dx, dw, db


def fused_linear_listnet_backward(
    do: torch.Tensor,
    dx: torch.Tensor,
    dw: torch.Tensor,
    db: torch.Tensor
):
    # If cross entropy is the last layer, do is 1.0. Skip the mul to save time
    if torch.ne(do, torch.tensor(1.0, device=do.device)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        N, H = dx.shape
        B = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        elementwise_mul_kernel[(triton.cdiv(N * H, B),)](
            x=dx,
            g=do,
            N=N*H,
            B=B,
            num_warps=32,
        )

        # handle dw
        if dw is not None:
            V, H = dw.shape
            elementwise_mul_kernel[(triton.cdiv(V * H, B),)](
                x=dw,
                g=do,
                N=V*H,
                B=B,
                num_warps=32,
            )

        if db is not None:
            V = db.shape[0]
            elementwise_mul_kernel[(triton.cdiv(V, B),)](
                x=db,
                g=do,
                N=V,
                B=B,
                num_warps=32,
            )
    return dx, dw, db


class FusedLinearListNetFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        targets: torch.Tensor,  # Float targets
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        ignore_index: int = -100,
        logit_scale: float = 1.0,
        num_chunks: int = 8,
        reduction: str = "mean"
    ):
        loss, dx, dw, db = fused_linear_listnet_forward(
            x, targets, weight, bias, ignore_index, 
            logit_scale, num_chunks, reduction
        )
        ctx.save_for_backward(dx, dw, db)
        return loss

    @staticmethod
    def backward(ctx, do):
        dx, dw, db = ctx.saved_tensors
        dx, dw, db = fused_linear_listnet_backward(do, dx, dw, db)
        return dx, None, dw, db, None, None, None, None


def fused_linear_listnet_loss(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    num_chunks: int = 8,
    reduction: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x (torch.Tensor): [batch_size * seq_len, hidden_size]
        target (torch.LongTensor): [batch_size * seq_len]
            where each value is in [0, vocab_size).
        weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        bias (Optional[torch.Tensor]): [vocab_size]
            where `vocab_size` is the number of classes.
        ignore_index: int.
            If target == ignore_index, the loss is set to 0.0.
        label_smoothing: float
        logit_scale: float
            A scaling factor applied to the logits. Default: 1.0
        num_chunks: int
            The number of chunks to split the input tensor into for processing.
            This can help optimize memory usage and computation speed.
            Default: 8
        reduction:
            Specifies the reduction to apply to the output: 'mean' | 'sum'.
            'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.
            Default: 'mean'.
    Returns:
        losses: [batch,], float
    """
    return FusedLinearListNetFunction.apply(
        x,
        target,
        weight,
        bias,
        ignore_index,
        logit_scale,
        num_chunks,
        reduction
    )


class FusedLinearListNetLoss(nn.Module):

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        num_chunks: int = 8,
        reduction: str = "mean"
    ):
        """
        Args:
            ignore_index: int.
                If target == ignore_index, the loss is set to 0.0.
            label_smoothing: float
            logit_scale: float
                A scaling factor applied to the logits. Default: 1.0
            num_chunks: int
                The number of chunks to split the input tensor into for processing.
                This can help optimize memory usage and computation speed.
                Default: 8
            reduction:
                Specifies the reduction to apply to the output: 'mean' | 'sum'.
                'mean': the weighted mean of the output is taken,
                'sum': the output will be summed.
                Default: 'mean'.
        """
        super().__init__()

        assert reduction in ["mean", "sum"], f"reduction: {reduction} is not supported"

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.num_chunks = num_chunks
        self.reduction = reduction

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, hidden_size]
            target (torch.LongTensor): [batch_size, seq_len]
                where each value is in [0, V).
            weight (torch.Tensor): [vocab_size, hidden_size]
                where `vocab_size` is the number of classes.
            bias (Optional[torch.Tensor]): [vocab_size]
                where `vocab_size` is the number of classes.
        Returns:
            loss
        """
        loss = fused_linear_listnet_loss(
            x.view(-1, x.shape[-1]),
            target.view(-1, target.shape[-1]),
            weight=weight,
            bias=bias,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale,
            num_chunks=self.num_chunks,
            reduction=self.reduction
        )
        return loss


class LinearLossParallel(ParallelStyle):
    def __init__(
        self,
        *,
        sequence_dim: int = 1,
        use_local_output: bool = False,
    ):
        super().__init__()

        self.sequence_sharding = (Shard(sequence_dim),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        x, target, weight, bias = inputs

        if not isinstance(x, DTensor):
            # assume the input passed in already sharded on the sequence dim and create the DTensor
            x = DTensor.from_local(x, device_mesh, sequence_sharding)
        if x.placements != sequence_sharding:
            x = x.redistribute(placements=sequence_sharding, async_op=True)
        if not isinstance(target, DTensor):
            target = DTensor.from_local(target, device_mesh, [Replicate()])
        if target.placements != sequence_sharding:
            target = target.redistribute(placements=sequence_sharding, async_op=True)

        if not isinstance(weight, DTensor):
            weight = DTensor.from_local(weight, device_mesh, [Replicate()])
        if weight.placements != [Replicate()]:
            # we replicate the weight/bias in FLCE
            weight = weight.redistribute(placements=[Replicate()], async_op=True)

        if bias is not None and not isinstance(bias, DTensor):
            bias = DTensor.from_local(bias, device_mesh, [Replicate()])
        if bias is not None and bias.placements != [Replicate()]:
            bias = bias.redistribute(placements=[Replicate()], async_op=True)

        return x.to_local(), target.to_local(), weight.to_local(), bias.to_local() if bias is not None else bias

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=partial(self._prepare_input_fn, self.sequence_sharding),
            output_fn=partial(self._prepare_output_fn, self.use_local_output)
        )
