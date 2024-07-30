from typing import Optional
import copy

import torch
from torch import nn


class ResidualStack(nn.Module):
    def __init__(
        self,
        residual_layer: nn.Module,
        n_layers: int,
        hidden_activation_fn: Optional[nn.Module] = None,
    ):
        hidden_activation_fn = (
            nn.ReLU()
            if hidden_activation_fn is None
            else copy.deepcopy(hidden_activation_fn)
        )
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [copy.deepcopy(residual_layer) for _ in range(n_layers)]
        )

        self.hidden_activation_fn = hidden_activation_fn

    def forward(self, x) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.hidden_activation_fn(x)
        
        return x
