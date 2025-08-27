import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_V': block_size_v}, num_warps=num_warp)
        for block_size_v in [256, 512, 1024, 2048]
        for num_warp in [1, 2, 4, 8]
    ],
    key=['V'],
)
@triton.jit
def _seq_to_top_kernel(
    seq_ptr,
    output_ptr,
    B,
    T_total,
    T,
    V,
    pad_token_id,
    window_size,
    T_val,
    stride_seq_b,
    stride_seq_t,
    stride_out_b,
    stride_out_t,
    stride_out_v,
    BLOCK_SIZE_V: tl.constexpr,
):
    b = tl.program_id(0)
    v_block = tl.program_id(1)
    
    v_start = v_block * BLOCK_SIZE_V
    v_end = tl.minimum(v_start + BLOCK_SIZE_V, V)
    v_idx = tl.arange(0, BLOCK_SIZE_V)
    v = v_start + v_idx
    mask = v < V
    
    next_occurrence = tl.full((BLOCK_SIZE_V,), T_val, dtype=tl.int64)
    
    for t in range(T_total - 1, -1, -1):
        token = tl.load(seq_ptr + b * stride_seq_b + t * stride_seq_t)
        
        token_valid = (token != pad_token_id) & (token >= 0) & (token < V)
        in_block = (token >= v_start) & (token < v_end)
        
        if token_valid:
            if in_block:
                local_v = token - v_start
                next_occurrence = tl.where(v_idx == local_v, t, next_occurrence)
        
        if t < T:
            distance = next_occurrence - t
            valid = (distance < window_size)
            value = tl.where(valid, window_size - distance, float('-inf'))
            
            output_offset = (
                b * stride_out_b +
                t * stride_out_t +
                v * stride_out_v
            )
            tl.store(output_ptr + output_offset, value, mask=mask)

def seq_to_top(
    seq: torch.Tensor, 
    vocab_size: int, 
    window_size: int,
    pad_token_id: int = -100
) -> torch.Tensor:
    """
    Triton-optimized top sequence processing with autotuned block size.
    
    :param seq: Input sequence of shape (B, T + window_size). If for training target, this input should be left shifted already.
    :param vocab_size: Size of the vocabulary
    :param window_size: How far ahead to look for next occurrences
    :param pad_token_id: Padding token ID
    :return: Tensor of shape (B, T, V) with window_size - distance for tokens in window, else -inf
    """
    B, T_total = seq.shape
    T = T_total - window_size
    
    assert T > 0, "T_total must be greater than window_size to produce valid output."
    
    output = torch.empty((B, T, vocab_size), device=seq.device, dtype=torch.float16)
    if not output.is_contiguous():
        output = output.contiguous()
    
    # Let autotune select the best BLOCK_SIZE_V based on vocab_size
    grid = (B, triton.cdiv(vocab_size, 128))  # Start with minimum block size
    
    _seq_to_top_kernel[grid](
        seq,
        output,
        B,
        T_total,
        T,
        vocab_size,
        pad_token_id,
        window_size,
        T_total,
        seq.stride(0),
        seq.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
    )
    
    return output