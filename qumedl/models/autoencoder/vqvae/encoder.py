import torch
from torch import nn


class EncoderBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Encoder must implement forward method.")

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self(inputs)
