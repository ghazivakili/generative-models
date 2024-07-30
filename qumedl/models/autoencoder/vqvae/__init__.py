import numpy as np
from .base import compute_vqvae_losses


def find_symmetric_vqvae_input_lengths(
    length: int, encoder_depth: int, stride: int = 2, search_space_size: int = 10
) -> list[int]:
    # NOTE: code assumes kernel_size = 3 and padding = 1
    # ol = (il + 2*p - ks) / stride + 1
    length_candidates = np.arange(length, length + search_space_size)

    numerator = length_candidates - 1

    # for our assumed kernel and padding for each incremental depth level
    # the equation is divided by the stride an extra time
    denominator = stride ** np.arange(1, encoder_depth)

    # matrix of lengths at all depth levels for each candidate length
    length_depth_grid = numerator[:, None] / denominator[None, :]

    # chose length such that at each depth the values in the grid are even
    return length_candidates[(length_depth_grid % 2 == 0).all(axis=1)]
