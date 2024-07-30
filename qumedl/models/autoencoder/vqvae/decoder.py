import torch
from torch import nn


class DecoderBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Decoder must implement forward method.")

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self(inputs)
