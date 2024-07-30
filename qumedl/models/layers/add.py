import torch
from torch import nn


class AddXY(nn.Module):
    """Add two tensors. Tensor shapes must be broadcastable."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class ProjectXAddY(nn.Module):
    """Project x to the shape of y and add them."""

    def __init__(
        self, input_dim: int, target_dim: int, activation: nn.Module | None = None
    ) -> None:
        """Initialize the layer.

        Args:
            target_dim (int): The target dimension of the projection.
        """
        super().__init__()
        self._projection = nn.Linear(input_dim, target_dim, bias=False)
        self._activation = activation or nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Project x to the shape of y and add them.

        Args:
            x (torch.Tensor): The tensor to project.
            y (torch.Tensor): The tensor to add to.

        Returns:
            torch.Tensor: The sum of x and y.
        """
        return self._activation(self._projection(x.float())) + y
