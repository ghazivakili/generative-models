import click
from trogon import tui

from qumedl.models.recurrent import PriorAssistedLSTM


@tui(command="train-ui", help="Train a model using terminal UI.")
@click.command()
@click.option("--batch_size", type=int)
@click.option("--embedding_dim", type=int)
@click.option(
    "--device",
    default="auto",
    help="Device to run on.",
    type=click.Choice(["auto", "cpu", "cuda"]),
)
def train(batch_size: int, embedding_dim: int, device: str = "auto"):
    print(batch_size)
    print(embedding_dim)
    print(device)


if __name__ == "__main__":
    train()
