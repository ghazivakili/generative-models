import json

import torch
from torch.utils.data import DataLoader
import tqdm
import wandb
import click
from trogon import tui

from qumedl.mol.encoding.selfies_ import Selfies
from qumedl.models.transformer.transformer import CausalMolTransformer
from qumedl.models.transformer.loss_functions import causal_transformer_compute_losses
from qumedl.training.collator import TensorBatchCollator
from qumedl.training.tensor_batch import TensorBatch
from qumedl.models.activations import NewGELU


@tui()
@click.command()
@click.option("--embedding-dim", type=int, default=256)
@click.option("--model-dim", type=int, default=256)
@click.option("--n-encoder-layers", type=int, default=8)
@click.option("--n-attn-heads", type=int, default=8)
@click.option("--dropout", type=float, default=0.1)
@click.option("--n-epochs", type=int, default=1)
@click.option("--gradient-accumulation-steps", type=int, default=1)
@click.option("--batch-size", type=int, default=64)
@click.option("--learning-rate", type=float, default=5e-3)
@click.option("--n-test-samples", type=int, default=100)
@click.option("--device", type=str, default=None)
@click.option("--random-seed", type=int, default=42)
@click.option(
    "--log-to-wandb", is_flag=True, show_default=True, type=bool, default=False
)
def main(
    embedding_dim: int,
    model_dim: int,
    n_encoder_layers: int,
    n_attn_heads: int,
    dropout: float,
    n_epochs: int,
    gradient_accumulation_steps: int,
    batch_size: int,
    learning_rate: float,
    n_test_samples: int,
    device: str | None = None,
    random_seed: int = 42,
    log_to_wandb: bool = False,
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

    model = CausalMolTransformer(
        vocab_size=selfies.n_tokens,
        embedding_dim=embedding_dim,
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
        run = wandb.init(project="qumedl")
        run.watch(model)
        run.config.update(
            {
                "embedding_dim": embedding_dim,
                "model_dim": model_dim,
                "n_encoder_layers": n_encoder_layers,
                "n_attn_heads": n_attn_heads,
                "dropout": dropout,
                "n_epochs": n_epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "n_test_samples": n_test_samples,
                "device": DEVICE,
                "random_seed": random_seed,
                "n_train_samples": len(selfies_dataset),
            }
        )
        print(run.config)

    # training loop
    for epoch in range(n_epochs):
        with tqdm.tqdm(total=len(selfies_dl)) as prog_bar:
            tensor_batch: TensorBatch
            for step, tensor_batch in enumerate(selfies_dl):
                tensor_batch.to(DEVICE)
                total_loss = causal_transformer_compute_losses(model, tensor_batch)

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

        prog_bar.set_description("Generating test molecules")

        # generate a few samples and save them as JSON locally and to WandB
        start_tokens = torch.full(
            (n_test_samples, 1),
            fill_value=selfies.start_index,
            device=DEVICE,
            dtype=torch.int,
        )

        generated = model.generate(start_tokens, seq_len=selfies.max_length)
        test_molecules = selfies.decode(generated.cpu().numpy())

        if run is not None:
            run.log({"test_molecules": test_molecules})

        with open(f"test_molecules-{epoch}.json", "w") as f:
            json.dump(test_molecules, f)

        prog_bar.set_description("Saving model state-dict locally.")
        torch.save(model.state_dict(), f"model-{epoch}.pt")

    if run is not None:
        run.finish()

    selfies.save_pickle(f"selfies-{epoch}.pkl")


if __name__ == "__main__":
    main()
