from typing import Optional

import torch
from torch import nn


class QuantizationLoss(nn.Module):
    def __init__(
        self, commitment_term: float = 0.25, loss_fn: Optional[nn.Module] = None
    ) -> None:
        """Computes loss between a set of input vectors and their quantized representations.
        This loss can be used with VQ-type models, such as the VQ-VAE.

        Args:
            commitment_term (float, optional): regulates the importance of the commitment loss term,
                which pushes encoder outputs to be closer to their quantized representations.
            loss_fn (LossFunction, optional): underlying loss function to use. If not specified,
                defaults to nn.MSELoss().
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        super().__init__()
        self.commitment_term = commitment_term
        self.loss_fn = loss_fn

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_vectors = inputs
        quantized_vectors = target
        encoding_latent_loss = self.loss_fn(quantized_vectors.detach(), input_vectors)
        quantized_latent_loss = self.loss_fn(quantized_vectors, input_vectors.detach())

        loss = quantized_latent_loss + self.commitment_term * encoding_latent_loss

        return loss
