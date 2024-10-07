import json
import os
import torch
from torch.utils.data import DataLoader
import tqdm
import wandb
import click
from trogon import tui
import pandas as pd
from qumedl.mol.encoding.selfies_ import Selfies
from qumedl.models.transformer.transformerCustom import CausalMolTransformerCustom
from qumedl.models.transformer.loss_functions import compute_transformer_loss
from qumedl.training.collator import TensorBatchCollator
from qumedl.training.tensor_batch import TensorBatch
from qumedl.models.activations import NewGELU
from orquestra.drug.discovery.docking.utils import process_molecule
from orquestra.drug.discovery.validator.filter_abstract import FilterAbstract
from orquestra.drug.metrics import MoleculeNovelty, get_diversity
from orquestra.drug.utils import ConditionFilters
import pickle
from datetime import datetime
import cloudpickle


class TartarusFilters(FilterAbstract):
    def apply(self, smile: str):
        _, status = process_molecule(smile)
        if status == "PASS":
            return True
        return False


def save_object(obj, filename):
    """Save a Python object to a file using pickle."""
    with open(filename, "wb") as file:  # Open the file in write-binary mode
        pickle.dump(obj, file)


def load_object(filename):
    """Load a Python object from a pickle file."""
    with open(filename, "rb") as file:  # Open the file in read-binary mode
        return pickle.load(file)


# save in file:
def save_obj(obj, file_path):
    with open(file_path, "wb") as f:
        r = cloudpickle.dump(obj, f)
    return r


def load_obj(file_path):
    with open(file_path, "rb") as f:
        obj = cloudpickle.load(f)
    return obj


def create_project_log_folder(project_name="pat"):
    # Generate a project name based on the current date
    current_date = datetime.now()
    # datetime.today().strftime("%Y_%d_%mT%H_%M_%S.%f")
    project_name = current_date.strftime(f"{project_name}_%Y-%m-%d_%H-%M-%S.%f")
    project_today = current_date.strftime(f"{project_name}_%Y-%m-%d")

    # Define the path for the logs directory
    logs_dir_path = "./logs"

    # Check if the logs directory exists, if not create it
    if not os.path.exists(logs_dir_path):
        os.makedirs(logs_dir_path)

    # Define the path for the new project directory within the logs folder
    project_dir_path = os.path.join(logs_dir_path, project_name)

    # Check if the project directory exists, if not create it
    if not os.path.exists(project_dir_path):
        os.makedirs(project_dir_path)

    print(f"Project log folder created at: {project_dir_path}")
    return (project_dir_path, project_name, project_today)


@tui()
@click.command()
@click.option("--embedding-dim", type=int, default=256)
@click.option("--model-dim", type=int, default=256)
@click.option("--n-encoder-layers", type=int, default=4)
@click.option("--n-attn-heads", type=int, default=8)
@click.option("--dropout", type=float, default=0.2)
@click.option("--n-epochs", type=int, default=1)
@click.option("--batch-size", type=int, default=64)
@click.option("--learning-rate", type=float, default=5e-3)
@click.option("--n-test-samples", type=int, default=100)
@click.option("--device", type=str, default=None)
@click.option("--gpu_address", type=int, default=1)
@click.option("--random-seed", type=int, default=42)
@click.option(
    "--full_dataset", is_flag=True, show_default=True, type=bool, default=False
)
@click.option("--gradient_accumulation_steps", type=int, default=5)
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
    gpu_address: int = 1,
    full_dataset: bool = False,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_address}"
    DEVICE = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # needs to be cuda on the cluster

    project_dir_path, project_name, project_today = create_project_log_folder()
    if full_dataset:
        dataset_name = (
            "/root/generative-models/scripts/data/docking_hill_climbing_0.csv"
        )
    else:
        dataset_name = (
            "/root/generative-models/scripts/data/docking_hill_climbing_0_sampled.csv"
        )

    # dataset_name = (
    # "/root/generative-models/scripts/data/docking_hill_climbing_0_sampled.csv"
    # )
    pickle_name = dataset_name.split(".")[0]
    if os.path.isfile(f"{pickle_name}.pkl"):
        selfies = load_object(f"{pickle_name}.pkl")
    else:
        selfies = Selfies.from_smiles_csv(dataset_name)
        save_object(selfies, f"{pickle_name}.pkl")

    smiles_dataset_df = pd.read_csv(dataset_name)
    smiles_dataset = smiles_dataset_df.smiles.to_list()
    filter_lists = [TartarusFilters()]  # ,SybaFilter()]
    weight_lists = [5.0]  # , 30.0]
    # novelity = MoleculeNovelty(smiles_dataset, threshold=0.6)
    filter = ConditionFilters(filter_lists=filter_lists, weight_lists=weight_lists)

    DEVICE = (
        device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )

    selfies_dataset = selfies.as_dataset(include_start_token=False)

    dl_shuffler = torch.Generator()
    dl_shuffler.manual_seed(random_seed)

    selfies_dl = DataLoader(
        selfies_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=dl_shuffler,
        collate_fn=TensorBatchCollator(),
    )

    model = CausalMolTransformerCustom(
        vocab_size=selfies.n_tokens,
        embedding_dim=embedding_dim,
        model_dim=embedding_dim,
        n_attn_heads=n_attn_heads,
        n_encoder_layers=n_encoder_layers,
        dropout=dropout,
        padding_token_idx=selfies.pad_index,
        max_seq_length=selfies.max_length,
    )
    print(DEVICE)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Non-trainable parameters
    non_trainable_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    run = None
    if log_to_wandb:
        run = wandb.init(
            project="Pat",
            name=f"{project_name}",
        )
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
                "n_params": trainable_params,
            }
        )
        print(run.config)

    # training loop
    for epoch in range(n_epochs):
        with tqdm.tqdm(total=len(selfies_dl)) as prog_bar:
            tensor_batch: TensorBatch
            for step, tensor_batch in enumerate(selfies_dl):
                tensor_batch.to(DEVICE)
                total_loss = compute_transformer_loss(model, tensor_batch)

                total_loss.backward()

                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step_losses = {"total_loss": total_loss.item()}

                prog_bar.set_postfix(step_losses)
                prog_bar.update()

                tensor_batch.to("cpu")
            print(f"epoch:{epoch}/{n_epochs}")
        if run is not None:
            wandb.log({"total_loss": total_loss.item()})
        prog_bar.set_description("Generating test molecules")

        # generate a few samples and save them as JSON locally and to WandB
        start_tokens = torch.full(
            (n_test_samples, 1),
            fill_value=selfies.start_index,
            device=DEVICE,
            dtype=torch.int,
        )

        generated = model.generate(start_tokens, max_new_tokens=selfies.max_length)
        test_molecules = selfies.decode(generated.cpu().numpy())
        # compute the metrics
        ligands_before = selfies.selfie_to_smiles(test_molecules)
        # novelity_rate = novelity.get_novelity_smiles(ligands_before)
        sr_rate = filter.get_validity_smiles(ligands_before)
        # diversity_rate = get_diversity(ligands_before)

        if run is not None:
            # run.log({"test_molecules": test_molecules})
            wandb.log(
                {
                    "sr_rate": sr_rate,
                    # "novelity_rate": novelity_rate,
                    # "diversity_rate": diversity_rate,
                }
            )

        with open(f"{project_dir_path}/test_molecules-{epoch}.json", "w") as f:
            json.dump(test_molecules, f)

        prog_bar.set_description("Saving model state-dict locally.")
        torch.save(model.state_dict(), f"{project_dir_path}/model-{epoch}.pt")

    if run is not None:
        run.finish()

    selfies.save_pickle(f"{project_dir_path}/selfies_obj.pkl")


if __name__ == "__main__":
    main()
