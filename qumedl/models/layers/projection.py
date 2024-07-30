import torch
from torch import nn


class Projection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        act: nn.Module,
    ) -> None:
        """Creates a projection block consisting of a linear layer followed by an activation function.

        Args:
            input_dim (int): input dimension to the linear layer.
            output_dim (int): output dimension of the linear layer.
            act (nn.Module): activation function applied to the output of the linear layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))
