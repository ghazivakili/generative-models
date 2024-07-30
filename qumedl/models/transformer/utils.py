import torch


def create_padding_mask(
    inputs: torch.Tensor, padding_token_idx: int, device: str | None = None
) -> torch.Tensor:
    """Creates a padding mask for a tensor."""
    b, seq_len = inputs.shape

    padding_mask = torch.zeros((b, seq_len), device=device)

    # positions with padding token become "-inf"
    padding_mask[padding_mask == padding_token_idx] = float("-inf")

    return padding_mask


def create_lookahead_mask(
    inputs: torch.Tensor, device: str | None = None
) -> torch.Tensor:
    """Creates a mask, preventing model from attending to "future" tokens."""
    b, seq_len = inputs.shape

    binary_mask = torch.tril(torch.ones(seq_len, seq_len))

    # lower diagonals become 0 and upper diagonals (not including main diagonal) become -inf
    return torch.log(binary_mask).to(device)
