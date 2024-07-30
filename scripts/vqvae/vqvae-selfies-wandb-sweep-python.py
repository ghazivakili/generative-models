import glob
import os
from typing import Optional

import torch
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"CUDA Visible Device Count: {torch.cuda.device_count()}")

import numpy as np
import selfies as sf
import wandb
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from qumedl.models.autoencoder.vqvae.base import (
    DecoderBase,
    DiscreteVQVAE,
    EncoderBase,
    VQVAEBase,
)
from qumedl.models.autoencoder.vqvae.quantizer import VectorQuantizer
from qumedl.models.autoencoder.vqvae.quantizer.loss_fn import QuantizationLoss
from qumedl.models.layers.residual import ResidualStack
from qumedl.models.recurrent.lstm import LSTM
from qumedl.mol.encoding.selfies_ import Selfies


class Conv1DResidualLayer(nn.Module):
    """Residual layer using 1D convolutions."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        hidden_activation_fn: Optional[nn.Module] = None,
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
        hidden_activation_fn: Optional[nn.Module] = None,
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
        hidden_activation_fn: Optional[nn.Module] = None,
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


def vqvae_single_train_step(
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


with open("./sweep-config.yaml") as file:
    sweep_config = yaml.load(file, Loader=yaml.FullLoader)

sweep_id = wandb.sweep(sweep=sweep_config, project="vqvae-dd")


def main():
    run = wandb.init()

    SEED = wandb.config.random_seed
    DEVICE = wandb.config.device

    selfies_config = wandb.config.selfies
    if selfies_config.get("pickle_path"):
        print("Loading selfies from pickle.")
        selfies = Selfies.from_pickle(selfies_config["pickle_path"])
    else:
        selfies = Selfies.from_smiles_csv(selfies_config["csv_path"])

    vqvae_config = wandb.config.vqvae
    torch.manual_seed(SEED)

    # potentially dangerous
    hidden_act_name: str = vqvae_config['hidden_act']
    leaky_relu_alpha: str = vqvae_config['leaky_relu_alpha']
    
    
    match hidden_act_name:
        case "relu":
            hidden_act = nn.ReLU(inplace=True)
        case "leaky_relu":
            hidden_act = nn.LeakyReLU(negative_slope=leaky_relu_alpha)
        case "selu":
            hidden_act = nn.SELU(inplace=True)
        case "gelu":
            hidden_act = nn.GELU()
        case "prelu":
            hidden_act = nn.PReLU()
        case _:
            print(f"Unatched case {hidden_act_name}, defaulting to nn.ReLU()")
            hidden_act = nn.ReLU()

    conv1d_vqvae = Conv1DVQVAE(
        vocab_size=selfies.n_tokens,
        embedding_dim=vqvae_config["embedding_dim"],
        hidden_dim=vqvae_config["hidden_dim"],
        residual_layer_hidden_dim=vqvae_config["residual_layer_hidden_dim"],
        n_codes=vqvae_config["n_codes"],
        hidden_activation_fn=hidden_act,
        pad_token=tuple([selfies.pad_token, selfies.pad_index]),
        output_activation_fn=nn.Identity(),
    ).to(DEVICE)

    print(f"Moving SELFIES to {DEVICE}")
    selfies_tensor = torch.from_numpy(selfies.asarray()).to(DEVICE)
    dataset = TensorDataset(selfies_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=vqvae_config["batch_size"],
        shuffle=True,
        generator=torch.Generator(),
    )

    vqvae_opt_config = vqvae_config["optimizer"]
    optimizer = torch.optim.AdamW(
        conv1d_vqvae.parameters(),
        lr=vqvae_opt_config["lr"],
        betas=(vqvae_opt_config["beta1"], vqvae_opt_config["beta2"]),
    )

    reconstruction_loss_fn = torch.nn.CrossEntropyLoss()
    quantization_loss_fn = QuantizationLoss()

    n_epochs = vqvae_config["n_epochs"]
    total_batches = n_epochs * len(dataloader)

    conv1d_vqvae.train()

    artifact = wandb.Artifact(f"checkpoints-run-{run.id}", type="checkpoint")
    with tqdm(total=total_batches) as pbar:
        for epoch in range(n_epochs):
            pbar.set_description(f"Epoch {epoch + 1}")
            for batch in dataloader:
                batch = batch[0].long()
                optimizer.zero_grad()
                rec_loss, q_loss = vqvae_single_train_step(
                    conv1d_vqvae, batch, reconstruction_loss_fn, quantization_loss_fn
                )

                total_loss = rec_loss + q_loss
                total_loss.backward()
                optimizer.step()

                step_losses = {
                    "loss": total_loss.item(),
                    "reconstruction_loss": rec_loss.item(),
                    "quantization_loss": q_loss.item(),
                }

                pbar_formatted_step_losses = {
                    k: round(v, 4) for k, v in step_losses.items()
                }

                pbar.set_postfix(pbar_formatted_step_losses)
                pbar.update()
                wandb.log(step_losses)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": conv1d_vqvae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    **step_losses,
                },
                f"./checkpoint-{epoch + 1}.pth",
            )

    artifact.add_file(f"./checkpoint-{epoch + 1}.pth")
    run.log_artifact(artifact)

    # remove all checkpoint files
    directory = "./"
    pattern = "checkpoint-*"

    # Construct full path pattern
    full_path_pattern = os.path.join(directory, pattern)

    # Find and delete files
    for file in glob.glob(full_path_pattern):
        os.remove(file)
        print(f"Deleted: {file}")


wandb.agent(sweep_id, function=main)
