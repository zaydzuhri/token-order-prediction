import torch
import torch.nn as nn
import torch.nn.functional as F

def seq_to_top(seq, vocab_size, window_size):
    """
    Calculates the "inverse" distance to the next occurrence of each token within a specified window.
    :param seq: Input sequence of shape (B, T + window_size). If for training target, this input should be left shifted already.
    :param vocab_size: Size of the vocabulary
    :param window_size: How far ahead to look for the next occurrence
    :return: Tensor of shape (B, T, V) with inverse distances (window_size - distance) for tokens in the window, else -inf
    """
    B, T_total = seq.shape
    T = T_total - window_size  # Output sequence length
    device = seq.device
    
    # Handle case where T_total < window_size (output is empty)
    if T <= 0:
        return torch.empty((B, 0, vocab_size), device=device, dtype=torch.float16)
    
    # Initialize output tensor for the first T positions
    y = torch.zeros((B, T, vocab_size), device=device, dtype=torch.float16)
    
    # Initialize next_occurrence with T_total (our "infinity")
    next_occurrence = torch.full((B, vocab_size), T_total, device=device, dtype=torch.long)
    
    # Handle OOV indices and create one_hot mask
    valid_mask = (seq >= 0) & (seq < vocab_size)
    adjusted_tokens = torch.where(valid_mask, seq, 0)
    one_hot = F.one_hot(adjusted_tokens, vocab_size).bool()
    one_hot = one_hot & valid_mask.unsqueeze(-1)  # (B, T_total, V)
    
    # Iterate backwards through time
    for t in reversed(range(T_total)):
        # Update next_occurrence with tokens at current position t
        current = one_hot[:, t]  # (B, V)
        next_occurrence = torch.where(current, t, next_occurrence)

        # Compute output for positions in [0, T-1]
        if t < T:
            distances = next_occurrence - t # (B, V)
            valid = distances <= window_size  # Tokens within the window
            # Set values: (window_size - distance) if valid, else -inf
            y[:, t, :] = torch.where(
                valid,
                (window_size - distances).to(y.dtype),
                float('-inf')
            )
    
    return y