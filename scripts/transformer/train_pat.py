import json
import requests

import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import wandb
import click
from trogon import tui

from qumedl.mol.encoding.selfies_ import Selfies
from qumedl.models.transformer.pat import CausalMolPAT
from qumedl.models.transformer.loss_functions import causal_transformer_compute_losses
from qumedl.training.collator import TensorBatchCollator
from qumedl.training.tensor_batch import TensorBatch
from qumedl.models.activations import NewGELU
from qumedl.models.priors import GaussianPrior


def generat_random_name(
    n: int = 2, prefix: str | None = None, suffix: str | None = None
) -> str:
    url = f"https://random-word-api.vercel.app/api?words={n}"

    words = list(requests.get(url).json())
    random_string = "-".join(words)

    if prefix is not None:
        random_string = f"{prefix}-{random_string}"

    if suffix is not None:
        random_string = f"{random_string}-{suffix}"

    return random_string


@tui()
@click.command()
@click.option("--embedding-dim", type=int, default=128)
@click.option("--model-dim", type=int, default=128)
@click.option("--n-encoder-layers", type=int, default=4)
@click.option("--n-attn-heads", type=int, default=8)
@click.option("--prior-dim", type=int, default=16)
@click.option("--dropout", type=float, default=0.1)
@click.option("--n-epochs", type=int, default=1)
@click.option("--gradient-accumulation-steps", type=int, default=1)
@click.option("--batch-size", type=int, default=32)
@click.option("--learning-rate", type=float, default=1e-3)
@click.option("--n-test-samples", type=int, default=1000)
@click.option("--temperature", type=float, default=1.0)
@click.option("--device", type=str, default=None)
@click.option("--random-seed", type=int, default=42)
@click.option("--log-to-wandb", type=bool, default=True)
def main(
    embedding_dim: int = 128,
    model_dim: int = 128,
    n_encoder_layers: int = 4,
    n_attn_heads: int = 8,
    prior_dim: int = 16,
    dropout: float = 0.1,
    n_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    n_test_samples: int = 1000,
    temperature: float = 1.0,
    device: str | None = None,
    random_seed: int = 42,
    log_to_wandb: bool = True,
):
    selfies = Selfies.from_smiles_csv(
        "/root/data/drug-discovery/1Mstoned_vsc_initial_dataset_insilico_chemistry42_filtered.csv"
    )
    DEVICE = (
        device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )

    selfies_dataset = selfies.as_dataset()

    dl_shuffler = torch.Generator()
    dl_shuffler.manual_seed(random_seed)

    selfies_dl = DataLoader(
        selfies_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=dl_shuffler,
        collate_fn=TensorBatchCollator(),
    )

    prior = GaussianPrior(dim=prior_dim)

    model = CausalMolPAT(
        vocab_size=selfies.n_tokens,
        embedding_dim=embedding_dim,
        prior_dim=prior.dim,
        model_dim=model_dim,
        n_attn_heads=n_attn_heads,
        n_encoder_layers=n_encoder_layers,
        hidden_act=NewGELU(),
        dropout=dropout,
        padding_token_idx=selfies.pad_index,
    )

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    run = None
    if log_to_wandb:
        run_name = generat_random_name(n=2, prefix="molpat")
        run = wandb.init(project="qumedl", name=run_name)
        run.watch(model)
        run.config.update(
            {
                "embedding_dim": embedding_dim,
                "model_dim": model_dim,
                "n_encoder_layers": n_encoder_layers,
                "n_attn_heads": n_attn_heads,
                "prior_dim": prior_dim,
                "dropout": dropout,
                "n_epochs": n_epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "n_test_samples": n_test_samples,
                "device": DEVICE,
                "random_seed": random_seed,
                "model_class": model.__class__.__name__,
                "prior_class": prior.__class__.__name__,
            }
        )
        print(run.config)

    # training loop
    for epoch in range(n_epochs):
        with tqdm.tqdm(total=len(selfies_dl)) as prog_bar:
            prog_bar.set_description(f"Training Epoch {epoch + 1} / {n_epochs}")
            tensor_batch: TensorBatch
            for step, tensor_batch in enumerate(selfies_dl):
                tensor_batch.to(DEVICE)
                prior_samples = prior.generate(tensor_batch.batch_size).to(DEVICE)
                total_loss = causal_transformer_compute_losses(
                    model, tensor_batch, prior_samples=prior_samples
                )

                total_loss.backward()

                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step_losses = {"total_loss": total_loss.item()}

                prog_bar.set_postfix(step_losses)
                prog_bar.update()

                if run is not None:
                    run.log(step_losses)

                tensor_batch.to("cpu")
                prior_samples.to("cpu")

        prog_bar.set_description("Generating test molecules")

        # generate a few samples and save them as JSON locally and to WandB
        test_prior_samples = prior.generate(n_test_samples).to(DEVICE)
        start_tokens = torch.full(
            (n_test_samples, 1),
            fill_value=selfies.start_index,
            device=DEVICE,
            dtype=torch.int,
        )

        generated = model.generate(
            start_tokens,
            test_prior_samples,
            max_new_tokens=selfies.max_length,
            temperature=temperature,
        )
        test_molecules = selfies.decode(generated.cpu().numpy())

        if run is not None:
            run.log({"test_molecules": test_molecules})

        with open(f"test_molecules-{epoch}.json", "w") as f:
            json.dump(test_molecules, f)

        prog_bar.set_description("Saving model state-dict locally.")
        torch.save(model.state_dict(), f"model-{epoch}.pt")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
