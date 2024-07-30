import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import json
from typing import Callable
from collections.abc import Sequence

import click
import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from wandb.sdk.wandb_run import Run
from trogon import tui
from tqdm import tqdm
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.markdown import Markdown

from qumedl.models.autoencoder.vqvae import (
    compute_vqvae_losses,
    find_symmetric_vqvae_input_lengths,
)
from qumedl.models.autoencoder.vqvae.base import VQVAEBase
from qumedl.models.autoencoder.vqvae.conv1d import Conv1DVQVAE
from qumedl.models.autoencoder.vqvae.quantizer.loss_fn import QuantizationLoss
from qumedl.mol.encoding.selfies_ import Selfies
from qumedl.models.transformer.transformer import CausalMolTransformer
from qumedl.models.transformer.loss_functions import compute_transformer_loss
from qumedl.training.utils import generate_random_name
from qumedl.training.collator import TensorBatchCollator
from qumedl.training.tensor_batch import TensorBatch
from qumedl.models.activations import NewGELU


console = Console()
rich_print = console.print


def print_dict_as_table(d: dict, title: str, style: str = "bold yellow"):
    """Prints a dictionary as a table.

    Args:
        d (dict): dictionary to print.
        title (str, optional): title to print above the table. Defaults to "".
    """

    table = Table(title=title)

    table.add_column("Key", style="cyan")
    table.add_column("Value", style="bright_green")

    for k, v in d.items():
        table.add_row(k, str(v))

    rich_print(table, style=style)


def save_checkpoints(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    directory: str,
    name: str,
    epoch: int,
    metrics: dict | None = None,
):
    name = f"{name}-{epoch}.pth"
    save_destination = os.path.join(directory, name)

    metrics = {} if metrics is None else metrics
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **metrics,
        },
        save_destination,
    )


@torch.no_grad()
def evaluate_vqvae(
    model: VQVAEBase,
    selfies: Selfies,
    batch_size: int,
    device: str,
) -> dict:
    """Runs a single VQ-VAE evaluation loop.

    Args:
        model (VQVAEBase): model to evaluate.
        selfies (Selfies): selfies dataset to evaluate on.
        batch_size (int): batch size to use for evaluation.
        device (str): device to use for evaluation.

    Returns:
        dict: a dictionary containing the averaged evaluation metrics.
    """
    # don't need start token for VQ-VAE
    eval_selfies_dataset = selfies.as_dataset(include_start_token=False)

    eval_data_loader = DataLoader(
        eval_selfies_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TensorBatchCollator(),
    )

    reconstruction_loss_fn = torch.nn.CrossEntropyLoss()
    quantization_loss_fn = QuantizationLoss()

    model.eval()

    metrics = {
        "vqvae/eval-loss": 0,
        "vqvae/eval-rec_loss": 0,
        "vqvae/eval-q_loss": 0,
        "n_steps": 0,
    }

    tensor_batch: TensorBatch
    for tensor_batch in eval_data_loader:
        # moves all tensors within batch to target device
        tensor_batch.to(device)

        rec_loss, q_loss = compute_vqvae_losses(
            model,
            tensor_batch.inputs.long(),
            reconstruction_loss_fn,
            quantization_loss_fn,
        )

        total_loss = rec_loss + q_loss

        # TODO[daniel]: compute perplexity
        step_losses = {
            "vqvae/eval-loss": total_loss.item(),
            "vqvae/eval-rec_loss": rec_loss.item(),
            "vqvae/eval-q_loss": q_loss.item(),
        }

        for key in step_losses:
            metrics[key] += step_losses[key]

        metrics["n_steps"] += 1

    # return average metrics
    n_steps = metrics.pop("n_steps")
    for key in metrics:
        metrics[key] /= n_steps

    return metrics


def round_dict_floats(d: dict, ndigits: int = 4) -> dict:
    """Rounds all float values in a dictionary to 4 decimal places.

    Args:
        d (dict): dictionary to round.
        ndigits (int, optional): number of digits to round to. Defaults to 4.

    Returns:
        dict: rounded dictionary.
    """

    return {k: round(v, ndigits) if isinstance(v, float) else v for k, v in d.items()}


def train_vqvae(
    model: VQVAEBase,
    selfies: Selfies | Sequence[Selfies, Selfies],
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    random_seed: int,
    device: str,
    wandb_run: Run | None = None,
):
    """Train VQ-VAE on selfies.

    Args:
        model (VQVAEBase): VQ-VAE model to train.
        selfies (Selfies | Sequence[Selfies, Selfies]): selfies to train on. If a single
            selfies object it will be considered as the training set. If a sequence of
            selfies objects, the first will be considered the training set and the second
            the validation set.
        batch_size (int): batch size to use for training.
        n_epochs (int): number of epochs to train for.
        learning_rate (float): learning rate to use for training.
        random_seed (int): random seed to use for training.
        device (str): device to use for training.
        wandb_run (Run | None, optional): wandb run to log to. Defaults to None.
    """
    # don't need start token for VQ-VAE training
    if isinstance(selfies, Selfies):
        validation_selfies = None

    elif isinstance(selfies, Sequence):
        validation_selfies = selfies[1]
        selfies = selfies[0]

    else:
        raise ValueError(
            "selfies must be a Selfies object or a sequence of Selfies objects"
        )

    train_selfies_dataset = selfies.as_dataset(include_start_token=False)

    shuffler = torch.Generator()
    shuffler.manual_seed(random_seed)

    train_data_loader = DataLoader(
        train_selfies_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffler,
        collate_fn=TensorBatchCollator(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.88, 0.995),  # TODO: expose as arguments
    )

    reconstruction_loss_fn = torch.nn.CrossEntropyLoss()
    quantization_loss_fn = QuantizationLoss()

    total_batches = n_epochs * len(train_data_loader)

    with tqdm(total=total_batches, desc="Training VQ-VAE") as pbar:
        for epoch in range(n_epochs):
            pbar.set_description(f"Training VQ-VAE: Epoch {epoch + 1}/{n_epochs}")
            tensor_batch: TensorBatch

            for tensor_batch in train_data_loader:
                # moves all tensors within batch to target device
                tensor_batch.to(device)

                rec_loss, q_loss = compute_vqvae_losses(
                    model,
                    tensor_batch.inputs.long(),
                    reconstruction_loss_fn,
                    quantization_loss_fn,
                )

                total_loss = rec_loss + q_loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # TODO: compute perplexity
                step_losses = {
                    "vqvae/train-loss": total_loss.item(),
                    "vqvae/train-rec_loss": rec_loss.item(),
                    "vqvae/train-q_loss": q_loss.item(),
                }

                pbar.set_postfix(step_losses)
                pbar.update()

                if wandb_run is not None:
                    wandb_run.log(step_losses)

            if validation_selfies is not None:
                val_metrics = evaluate_vqvae(
                    model, validation_selfies, batch_size, device
                )

                pbar.set_postfix({**step_losses, **val_metrics})

                if wandb_run is not None:
                    wandb_run.log(val_metrics)


@torch.no_grad()
def evaluate_transformer(
    model: CausalMolTransformer,
    dataset: TensorDataset,
    batch_size: int,
    device: str,
    wandb_run: Run | None = None,
) -> dict:
    """Evaluate the transformer model on the validation set."""

    eval_data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TensorBatchCollator(),
    )

    model.eval()

    eval_metrics = {
        "transformer/eval-loss": 0,
        "transformer/eval-perplexity": 0,
        "n_steps": 0,
    }

    tensor_batch: TensorBatch
    for step, tensor_batch in enumerate(eval_data_loader):
        tensor_batch.inputs = tensor_batch.inputs.long()
        tensor_batch.to(device)
        loss = compute_transformer_loss(model, tensor_batch)

        step_metrics = {
            "transformer/eval-loss": loss.item(),
            "transformer/eval-perplexity": torch.exp(loss).item(),
        }

        if wandb_run is not None:
            wandb_run.log(step_metrics)

        for metric in step_metrics:
            if metric not in eval_metrics:
                eval_metrics[metric] = 0

            eval_metrics[metric] += step_metrics[metric]

        eval_metrics["n_steps"] += 1

    n_steps = eval_metrics.pop("n_steps")
    for metric in eval_metrics:
        eval_metrics[metric] /= n_steps

    return eval_metrics


def train_transformer(
    model: CausalMolTransformer,
    dataset: TensorDataset | Sequence[TensorDataset, TensorDataset],
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    random_seed: int,
    device: str,
    gradient_accumulation_steps: int = 1,
    wandb_run: Run | None = None,
) -> None:
    """Train the latent transformer model and optionally evaluate it on the validation set.

    Args:
        model (CausalMolTransformer): The model to train.
        dataset (TensorDataset | Sequence[TensorDataset, TensorDataset]): The dataset to train on.
            If a sequence is passed, the first element is used for training and the second for validation.
        batch_size (int): The batch size to use for training.
        n_epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use for training.
        random_seed (int): The random seed to use for training.
        device (str): The device to use for training.
        gradient_accumulation_steps (int, optional): The number of gradient accumulation steps to use. Defaults to 1.
        wandb_run (Run, optional): The wandb run to use for logging. Defaults to None.
    """

    if isinstance(dataset, Sequence):
        dataset, validation_dataset = dataset
    elif isinstance(dataset, TensorDataset):
        validation_dataset = None
    else:
        raise ValueError(f"Invalid dataset type: {type(dataset)}")

    shuffler = torch.Generator()
    shuffler.manual_seed(random_seed)

    train_data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffler,
        collate_fn=TensorBatchCollator(),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    total_batches = n_epochs * len(train_data_loader)

    model.train()
    model.to(device)

    # training loop
    with tqdm(total=total_batches, desc="Training Latent Transformer") as pbar:
        for epoch in range(n_epochs):
            pbar.set_description(f"Training Transformer:  Epoch {epoch + 1}/{n_epochs}")

            eval_metrics = {}
            tensor_batch: TensorBatch
            for step, tensor_batch in enumerate(train_data_loader):
                tensor_batch.inputs = tensor_batch.inputs.long()
                tensor_batch.to(device)
                loss = compute_transformer_loss(model, tensor_batch)

                loss.backward()

                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step_metrics = round_dict_floats(
                    {
                        "transformer/train-loss": loss.item(),
                        "transformer/train-perplexity": torch.exp(loss).item(),
                    }
                )

                pbar.set_postfix(step_metrics)
                pbar.update()

                if wandb_run is not None:
                    wandb_run.log(step_metrics)

            if validation_dataset is not None:
                eval_metrics = evaluate_transformer(
                    model, validation_dataset, batch_size, device, wandb_run
                )
                pbar.set_postfix({**step_metrics, **eval_metrics})

                if wandb_run is not None:
                    wandb_run.log(eval_metrics)

            # return model to train mode which may have been changed during evaluation
            model.train()


@tui()
@click.command()
@click.option("--vq-embedding-dim", type=int, default=256)
@click.option("--vq-hidden-dim", type=int, default=128)
@click.option("--vq-n-codes", type=int, default=512)
@click.option(
    "--vq-hidden-activation-fn",
    type=str,
    default="nn.LeakyReLU(negative_slope=0.001)",
    help="String specifying the initialization of the VQ-VAE hidden activation function. Will be evaluated using ``eval``.",
)
@click.option("--vq-n-epochs", type=int, default=20)
@click.option("--vq-batch-size", type=int, default=256)
@click.option(
    "--vq-learning-rate",
    type=float,
    default=5e-3,
    help="Lower learning rate may be better for the VQ-VAE. Varies with dataset.",
)
@click.option("--tr-embedding-dim", type=int, default=256)
@click.option("--tr-model-dim", type=int, default=128)
@click.option("--tr-n-encoder-layers", type=int, default=8)
@click.option("--tr-n-attn-heads", type=int, default=8)
@click.option(
    "--tr-hidden-activation-fn",
    type=str,
    default="NewGELU()",
    help="String specifying the initialization of the latent Transformer hidden activation function. Will be evaluated using ``eval``.",
)
@click.option("--tr-dropout", type=float, default=0.1)
@click.option("--tr-n-epochs", type=int, default=10)
@click.option("--tr-gradient-accumulation-steps", type=int, default=1)
@click.option("--tr-batch-size", type=int, default=256)
@click.option("--tr-learning-rate", type=float, default=5e-3)
@click.option("--tr-n-test-samples", type=int, default=50)
@click.option("--temperature", type=float, default=1.0)
@click.option("--device", type=str, default=None)
@click.option("--random-seed", type=int, default=42)
@click.option(
    "--dataset-size",
    type=click.Choice(["1K", "10K", "100K", "1M"], case_sensitive=False),
    default="100K",
    help="Size of the dataset to use. 1K is the smallest, 1M is the full dataset.",
)
@click.option("--log-to-wandb", type=bool, default=True)
@click.option("--wandb-project", type=str, default="qumedl")
def main(
    vq_embedding_dim: int,
    vq_hidden_dim: int,
    vq_n_codes: int,
    vq_hidden_activation_fn: str | nn.Module | Callable[[torch.Tensor], torch.Tensor],
    vq_n_epochs: int,
    vq_batch_size: int,
    vq_learning_rate: float,
    tr_embedding_dim: int,
    tr_model_dim: int,
    tr_n_encoder_layers: int,
    tr_n_attn_heads: int,
    tr_hidden_activation_fn: str | nn.Module | Callable[[torch.Tensor], torch.Tensor],
    tr_dropout: float,
    tr_n_epochs: int,
    tr_gradient_accumulation_steps: int,
    tr_batch_size: int,
    tr_learning_rate: float,
    tr_n_test_samples: int,
    temperature: float,
    device: str,
    random_seed: int,
    dataset_size: str,
    log_to_wandb: bool,
    wandb_project: str,
):
    torch.manual_seed(random_seed)
    device = (
        device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )

    run: Run | None = None
    if log_to_wandb:
        run_name = generate_random_name(2, prefix="vqvae+transformer")
        run = wandb.init(wandb_project, name=run_name)
        run.config.update(
            {
                "data": {
                    "path": f"/root/data/drug-discovery/{dataset_size}stoned_vsc_initial_dataset_insilico_chemistry42_filtered.csv"
                },
                "transformer": {
                    "embedding_dim": tr_embedding_dim,
                    "model_dim": tr_model_dim,
                    "n_encoder_layers": tr_n_encoder_layers,
                    "n_attn_heads": tr_n_attn_heads,
                    "dropout": tr_dropout,
                    "n_epochs": tr_n_epochs,
                    "gradient_accumulation_steps": tr_gradient_accumulation_steps,
                    "batch_size": tr_batch_size,
                    "learning_rate": tr_learning_rate,
                    "n_test_samples": tr_n_test_samples,
                    "temperature": temperature,
                    "device": device,
                    "random_seed": random_seed,
                    "hidden_activation_fn": tr_hidden_activation_fn,
                },
                "vqvae": {
                    "embedding_dim": vq_embedding_dim,
                    "hidden_activation_fn": str(vq_hidden_activation_fn),
                    "hidden_dim": vq_hidden_dim,
                    "batch_size": vq_batch_size,
                    "hidden_activation_fn": vq_hidden_activation_fn,
                    "learning_rate": vq_learning_rate,
                    "n_codes": vq_n_codes,
                    "n_epochs": vq_n_epochs,
                },
            }
        )

        print_dict_as_table(run.config, title="Experiment Configuration")

        if not os.path.exists(run_name):
            os.makedirs(run_name)

    # hardcoded max_length is the length of the longest sequence in the 1M dataset
    selfies = Selfies.from_smiles_csv(run.config["data"]["path"])

    # length of samples such that samples reconstructed with the Conv1D VQ-VAE keep the same length as the inputs
    symmetric_lengths = find_symmetric_vqvae_input_lengths(
        selfies.max_length, search_space_size=20, encoder_depth=3
    )
    selfies.max_length = symmetric_lengths[0]
    selfies, val_selfies = selfies.train_test_split(
        train_ratio=0.95, random_seed=random_seed
    )

    vqvae = Conv1DVQVAE(
        vocab_size=selfies.n_tokens,
        embedding_dim=vq_embedding_dim,
        hidden_dim=vq_hidden_dim,
        residual_layer_hidden_dim=vq_hidden_dim,
        n_codes=vq_n_codes,
        hidden_activation_fn=eval(vq_hidden_activation_fn)
        if isinstance(vq_hidden_activation_fn, str)
        else vq_hidden_activation_fn,
        pad_token=(selfies.pad_token, selfies.pad_index),
        output_activation_fn=nn.Identity(),
    ).to(device)

    if run is not None:
        run.config["vqvae"]["model_class"] = vqvae.__class__.__name__

    # vq-vae checkpoints will be saved locally to disk for every epoch
    train_vqvae(
        vqvae,
        (selfies, val_selfies),
        vq_batch_size,
        vq_n_epochs,
        vq_learning_rate,
        random_seed,
        device,
        wandb_run=run,
    )

    # vocab of the transformer are all the possible codes of the VQ-VAE as the Tranformer is trained on latent space sequences
    transformer = CausalMolTransformer(
        vocab_size=vqvae.n_codes,
        embedding_dim=tr_embedding_dim,
        model_dim=tr_embedding_dim,
        n_attn_heads=tr_n_attn_heads,
        n_encoder_layers=tr_n_encoder_layers,
        hidden_act=eval(tr_hidden_activation_fn)
        if isinstance(tr_hidden_activation_fn, str)
        else tr_hidden_activation_fn,
        dropout=0.1,
        # FIXME: why no padding token?
        padding_token_idx=None,
    ).to(device)

    if run is not None:
        run.config["transformer"]["model_class"] = transformer.__class__.__name__

    # NOTE: latent sequences will not start with the same token
    # NOTE: switch to using same start token
    latent_train_selfies = vqvae.batch_encode_index(
        selfies.as_tensor(include_start_token=False).to(device), batch_size=512
    )

    latent_val_selfies = vqvae.batch_encode_index(
        val_selfies.as_tensor(include_start_token=False).to(device), batch_size=512
    )

    train_transformer(
        transformer,
        dataset=(
            TensorDataset(latent_train_selfies),
            TensorDataset(latent_val_selfies),
        ),
        batch_size=tr_batch_size,
        n_epochs=tr_n_epochs,
        learning_rate=tr_learning_rate,
        random_seed=random_seed,
        device=device,
        gradient_accumulation_steps=tr_gradient_accumulation_steps,
        wandb_run=run,
    )

    # NOTE: start tokens are sampled randomly
    # need to figure out way to have consistent start token
    start_tokens = torch.randint(
        low=0,
        high=vqvae.n_codes,
        size=(tr_n_test_samples, 1),
        device=device,
        dtype=torch.int,
    )

    transformer.eval()
    vqvae.eval()

    rich_print(Markdown("### Generating and Saving Samples"))

    with torch.no_grad():
        # generate latent samples using transformer
        # shape (n_test_samples, max_new_tokens)
        sampled_latent_sequences = transformer.generate(
            start_tokens,
            # TODO: 54 is the observed latent squence length -> result length 127 (max_new_tokens) + 1 (initial)
            max_new_tokens=54,
            temperature=temperature,
        )

        # shape (n_test_samples, code_dim, max_new_tokens + 1)
        latent_sampled_codes = vqvae.codes_from_indices(sampled_latent_sequences)

        encoded_generated_distr = vqvae.decode(latent_sampled_codes)

    # greedy generation
    encoded_generated_molecules = encoded_generated_distr.argmax(dim=1).cpu().numpy()

    # decode into molecules
    generated_molecules = selfies.decode(
        encoded_generated_molecules, stop_token_id=selfies.pad_index
    )

    save_samples_to = "vqvae+transformer-generated-samples.json"
    if run is not None:
        save_samples_to = os.path.join(run.name, save_samples_to)

    with open(save_samples_to, "w") as f:
        json.dump(generated_molecules, f)

    if run is not None:
        run.log({"generated_molecules": generated_molecules})
        run.finish()


if __name__ == "__main__":
    main()
