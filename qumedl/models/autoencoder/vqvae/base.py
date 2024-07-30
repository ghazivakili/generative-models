from typing import Optional, Tuple

import torch
from torch import nn
from tqdm import trange

from .quantizer import VectorQuantizer


class EncoderBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Encoder must implement forward method.")

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self(inputs)


class DecoderBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Decoder must implement forward method.")

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self(inputs)


class VQVAEBase(nn.Module):
    """Base class for VQ-VAE models."""

    def __init__(
        self,
        encoder: EncoderBase,
        quantizer: VectorQuantizer,
        decoder: DecoderBase,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(inputs)
        quantized, indices = self.quantizer(encoded)
        decoded = self.decoder(quantized)
        return decoded, indices

    @property
    def n_codes(self) -> int:
        """Returns the number of code vectors in the codebook."""
        return self.quantizer.n_codes

    @property
    def code_dim(self) -> int:
        """Returns the dimension of the code vectors in the codebook."""
        return self.quantizer.code_dim

    def get_codebook(self) -> torch.Tensor:
        """Returns all of the code vectors in the codebook."""
        return self.quantizer.codebook.weight.data.detach()

    def codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Returns a tensor of code vectors at the given indices in the codebook.

        Args:
            indices (torch.Tensor): a tensor of shape (n_samples, n_code_vectors),
                where each element is an index of a desired code vector in the codebook.

        Returns:
            torch.Tensor: tensor of shape (n_samples, code_vector_dim, n_code_vectors).
        """

        codebook = self.get_codebook()
        indices = indices.to(codebook.device)
        n_samples, n_vectors = indices.shape

        # use (B, code_dim, n_vecs) shape to match the output format of the `quantize`
        # method
        output = torch.zeros(
            (n_samples, self.code_dim, n_vectors), device=codebook.device
        )

        # torch.index_select does not support 2D index tensors
        # https://pytorch.org/docs/stable/generated/torch.index_select.html
        for row, row_indices in enumerate(indices):
            # has shape (n_code_vectors, code_vector_dim)
            selected_codes = torch.index_select(codebook, dim=0, index=row_indices)

            output[row, :, :] = selected_codes.permute(1, 0)

        return output

    def random_codes(self, n_codes: int, seed: Optional[int] = None) -> torch.Tensor:
        """Returns a tensor of randomly sampled vectors from the codebook.

        Args:
            n_codes (int): number of code vectors to select.
            seed (Optional[int], optional): random seed. Defaults to None.

        Returns:
            torch.Tensor: tensor of shape (n_code_vectors, code_vector_dim).
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # pylint: disable=no-member
        code_vector_indices = torch.randint(low=0, high=self.n_codes, size=(n_codes,))
        code_vectors = self.codes_from_indices(code_vector_indices)

        return code_vectors

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor into a tensor in the autoencoder's latent space.

        Args:
            inputs (torch.Tensor): input tensor.
        """
        return self.encoder(inputs)

    def decode(self, code_vectors: torch.Tensor) -> torch.Tensor:
        """Decodes the given code vectors into a tensor in the autoencoder's data space.

        Args:
            code_vectors (torch.Tensor): tensor of code vectors to decode.
        """
        return self.decoder(code_vectors)

    def encode_index(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor into a tensor of indices of code vectors in the codebook.

        Args:
            inputs (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: tensor of indices of code vectors in the codebook.
        """
        encoded = self.encode(inputs)
        _, indices = self.quantizer(encoded)
        return indices

    def batch_encode_index(self, inputs: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Encodes data in the latent space of the VQ-VAE, but breaks the task down to
        batches of `batch_size`.
        
        This can be useful when the full inputs are too large to be passed to the VQ-VAE at once.
        """
        encoded_data: torch.Tensor | None = None

        for i in trange(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            # encode every sample and conver into sequences
            encoded_batch = self.encode_index(batch)

            if encoded_data is None:
                # assume every sequence will have the same length in latent space
                _, latent_seq_len = encoded_batch.shape
                encoded_data = torch.zeros((inputs.shape[0], latent_seq_len))

            encoded_data[i : i + batch_size] = encoded_batch

        return encoded_data

    def encode_quantize(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor and quantizes the resulting code vectors.

        Args:
            inputs (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: tensor of quantized code vectors.
        """
        encoded = self.encode(inputs)
        quantized, _ = self.quantizer(encoded)
        return quantized


class DiscreteVQVAE(VQVAEBase):
    """Base class for VQ-VAE models that take discrete values as inputs."""

    def __init__(
        self,
        input_embedding: nn.Module,
        encoder: EncoderBase,
        quantizer: VectorQuantizer,
        decoder: DecoderBase,
    ) -> None:
        super().__init__(encoder, quantizer, decoder)
        self.input_embedding = input_embedding

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.input_embedding(inputs)
        return super().__call__(inputs)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor into a tensor in the autoencoder's latent space.

        Args:
            inputs (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: tensor in the autoencoder's latent space.
        """
        inputs = self.input_embedding(inputs)
        return super().encode(inputs)

    def encode_index(self, inputs: torch.Tensor, one_hot: bool = False) -> torch.Tensor:
        """Encodes the input tensor into a tensor of indices of code vectors in the codebook.

        Args:
            inputs (torch.Tensor): input tensor.
            one_hot (bool): whether to return a one-hot encoding of the indices. Defaults to False.

        Returns:
            torch.Tensor: tensor of indices of code vectors in the codebook.
        """
        encoded = self.encode(inputs)
        _, indices = self.quantizer(encoded)

        if one_hot:
            return indices

        return indices.argmax(dim=1).view(inputs.shape[0], -1)

    def encode_quantize(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor and quantizes the resulting code vectors.

        Args:
            inputs (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: tensor of quantized code vectors.
        """
        encoded = self.encode(inputs)
        quantized, _ = self.quantizer(encoded)
        return quantized


def compute_vqvae_losses(
    model: VQVAEBase,
    data: torch.Tensor,
    rec_loss_function: nn.Module,
    q_loss_function: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Progresses VQ-VAE through a single training step.

    Args:
        model (VQVAEBase): model to train.
        data (torch.Tensor): data to train model on.
        rec_loss_function (nn.Module): function to compute the reconstruction loss.
        q_loss_function (nn.Module): function to compute the quantization loss.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: a tuple containing the reconstruction loss and quantization loss,
            with gradients attached.
    """
    encoded = model.encode(data)

    quantized, _ = model.quantizer.quantize(encoded)

    # compute quantization loss here since next step is to copy gradient from
    # quantized to encoded
    quantization_loss = q_loss_function(encoded, quantized)

    # this trick creates a connection in the computation graph between the quantized
    # tensor and the encoded tensor otherwise we cannot backpropagate through the
    # quantization step
    quantized = encoded + (quantized - encoded).detach()

    reconstruction = model.decode(quantized)

    reconstruction_loss = rec_loss_function(reconstruction, data)

    return reconstruction_loss, quantization_loss
