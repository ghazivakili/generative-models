import torch
from einops.layers.torch import Rearrange
from torch import nn

from qumedl.models.autoencoder.vqvae.base import DecoderBase, DiscreteVQVAE, EncoderBase
from qumedl.models.autoencoder.vqvae.quantizer import VectorQuantizer
from qumedl.models.layers.residual import ResidualStack


class Conv1DResidualLayer(nn.Module):
    """Residual layer using 1D convolutions."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        hidden_activation_fn: nn.Module | None = None,
    ):
        """Initializes a new Conv1DResidualLayer instance.

        Args:
            in_channels (int): number of input channels
            hidden_dim (int): number of hidden channels
            hidden_activation_fn (Optional[nn.Module], optional): activation function to use. Defaults to None.
                Is applied between the convolutional layers. Defaults to ReLU.
        """
        hidden_activation_fn = nn.ReLU(inplace=True) or hidden_activation_fn

        super().__init__()
        self._block = nn.Sequential(
            hidden_activation_fn,
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            hidden_activation_fn,
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x) -> torch.Tensor:
        return x + self._block(x)


class Conv1DEncoder(EncoderBase):
    """1D convolutional encoder for the VQ-VAE.
    This module is built up using a series of 1D convolutional layers, with a residual
    stack in between.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        n_residual_layers: int,
        residual_layer_hidden_dim: int,
        hidden_activation_fn: nn.Module | None = None,
    ):
        """Convolutional Encoder for VQ-VAE.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            hidden_dim (int): dimension of hidden layers.
            n_residual_layers (int): number of residual layers.
            residual_layer_hidden_dim (int): hidden dimension of each residual layer.
            hidden_activation_fn (nn.Module, optional): activation function between
                hidden layers. If None, defaults to nn.ReLU().
        """
        hidden_activation_fn = nn.ReLU(inplace=True) or hidden_activation_fn

        super().__init__()

        self.conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_dim // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_dim // 2,
            out_channels=hidden_dim,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        residual_layer = Conv1DResidualLayer(
            in_channels=hidden_dim,
            hidden_dim=residual_layer_hidden_dim,
            hidden_activation_fn=hidden_activation_fn,
        )
        self.residual_stack = ResidualStack(
            residual_layer=residual_layer,
            n_layers=n_residual_layers,
            hidden_activation_fn=hidden_activation_fn,
        )
        self.pre_quantization_conv = nn.Conv1d(
            in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, stride=1
        )
        self.hidden_activation_fn = hidden_activation_fn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(inputs)
        x = self.hidden_activation_fn(x)

        x = self.conv_2(x)
        x = self.hidden_activation_fn(x)

        x = self.conv_3(x)
        x = self.residual_stack(x)

        return self.pre_quantization_conv(x)


class Conv1DDecoder(DecoderBase):
    """1D convolutional decoder for the VQ-VAE.
    This module is built up using a series of 1D convolutional transpose layers, with a
    residual stack in between.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        n_residual_layers: int,
        residual_layer_hidden_dim: int,
        output_activation_fn: nn.Module,
        hidden_activation_fn: nn.Module | None = None,
    ):
        """Convolutional decoder for the VQ-VAE.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            hidden_dim (int): dimension of hidden layers.
            n_residual_layers (int): number of residual layers.
            residual_layer_hidden_dim (int): hidden dimension of each residual layer.
            output_activation_fn (nn.Module): activation function applied to the final
                utputs.
            hidden_activation_fn (nn.Module, optional): activation function between
                hidden layers. If None, defaults to nn.ReLU().
        """
        hidden_activation_fn = (
            nn.ReLU() if hidden_activation_fn is None else hidden_activation_fn
        )
        super().__init__()

        self.conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        residual_layer = Conv1DResidualLayer(
            in_channels=hidden_dim,
            hidden_dim=residual_layer_hidden_dim,
            hidden_activation_fn=hidden_activation_fn,
        )

        self.residual_stack = ResidualStack(
            residual_layer=residual_layer,
            n_layers=n_residual_layers,
            hidden_activation_fn=hidden_activation_fn,
        )

        self.conv_trans_1 = nn.ConvTranspose1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            kernel_size=5,
            stride=2,
            padding=1,
        )

        self.conv_trans_2 = nn.ConvTranspose1d(
            in_channels=hidden_dim // 2,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # TODO: hardcoding for testing. Need to figure out how to make this dynamic
        self.match_seq_len_linear = nn.Identity()
        self.hidden_activation_fn = hidden_activation_fn
        self.output_activation_fn = output_activation_fn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder.

        Args:
            inputs (torch.Tensor): an input tensor of shape (batch_size, features, seq_len)

        Returns:
            torch.Tensor: _description_
        """
        x = self.conv_1(inputs)
        x = self.residual_stack(x)
        x = self.conv_trans_1(x)
        x = self.hidden_activation_fn(x)

        # has shape (batch_size, out_features, ?)
        x = self.conv_trans_2(x)
        x = self.hidden_activation_fn(x)
        x = self.match_seq_len_linear(x)

        return self.output_activation_fn(x)


class Conv1DVQVAE(DiscreteVQVAE):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        pad_token: tuple[str, int],
        hidden_dim: int = 128,
        n_residual_layers: int = 2,
        residual_layer_hidden_dim: int = 32,
        n_codes: int = 512,
        code_dim: int = 64,
        hidden_activation_fn: nn.Module | None = None,
        output_activation_fn: nn.Module | None = None,
    ):
        """Initializes the VQ-VAE model.

        Args:
            n_channels (int): number of channels in the input tensors.
            vcab_size (int): size of the vocabulary, including start and padding tokens.
            embedding_dim (int): dimension of the embeddings.
            pad_token (tuple(str, int)): tuple containing the name and index of the pad token.
            hidden_dim (int, optional): hidden dimension of the model. Defaults to 128.
            n_residual_layers (int, optional): number of layers in the residual stack.
                Defaults to 2.
            residual_layer_hidden_dim (int, optional): hidden dimension of each residual
                layer. Defaults to 32.
            n_codes (int, optional): number of 'codes' in the codebook. Defaults to 512.
            code_dim (int, optional): dimension of each code in the codebook. Defaults
                to 64.
            hidden_activation_fn (nn.Module, optional): activation function used in
                hidden layers. If None defaults to nn.ReLU().
            output_activation_fn (nn.Module): activation function applied to the final
                outputs of the decoder.
        """
        # encoder outputs a tensor of shape (batch_size, out_channels, seq_len)
        # quantizer expects tensor of shape (batch_size, seq_len,
        # out_channels=embedding_dim)
        out_channels = code_dim
        permute_order = [0, 2, 1]

        hidden_activation_fn = nn.ReLU(inplace=True) or hidden_activation_fn
        output_activation_fn = nn.Identity() or output_activation_fn

        encoder = Conv1DEncoder(
            embedding_dim,
            out_channels,
            hidden_dim,
            n_residual_layers,
            residual_layer_hidden_dim,
            hidden_activation_fn,
        )
        quantizer = VectorQuantizer(
            n_codes=n_codes, code_dim=code_dim, permute_order=permute_order
        )
        decoder = Conv1DDecoder(
            code_dim,
            vocab_size,
            hidden_dim,
            n_residual_layers,
            residual_layer_hidden_dim,
            output_activation_fn,
            hidden_activation_fn,
        )

        input_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=pad_token[1],
            ),
            Rearrange("b s e -> b e s"),
        )

        super().__init__(input_embedding, encoder, quantizer, decoder)

    @staticmethod
    def _debug_forward_hook(
        module: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> None:
        if isinstance(inputs, (list, tuple)):
            inputs_shape = [i.shape for i in inputs]
        else:
            inputs_shape = inputs.shape

        if isinstance(outputs, (list, tuple)):
            outputs_shape = [o.shape for o in outputs]
        else:
            outputs_shape = outputs.shape

        print(f"Module={module._get_name()}: {inputs_shape} -> {outputs_shape}")

    @staticmethod
    def _is_leaf_module(module: nn.Module) -> bool:
        return len(list(module.children())) == 0

    def enable_debug_hooks(self) -> None:
        for module_name, module in self.named_modules():
            if self._is_leaf_module(module):
                module.register_forward_hook(self._debug_forward_hook)

    def disable_debug_hooks(self) -> None:
        for module_name, module in self.named_modules():
            module._forward_hooks.clear()
