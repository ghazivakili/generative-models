import json

import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import tqdm
import numpy as np
from qumedl.mol.encoding.selfies_ import Selfies
from qumedl.models.transformer.pat import CausalMolPAT
from qumedl.models.transformer.loss_functions import (
    compute_transformer_loss,
    compute_transformer_loss_vlad,
)
from qumedl.training.collator import TensorBatchCollator
from qumedl.training.tensor_batch import TensorBatch
from qumedl.models.activations import NewGELU
from qumedl.models.priors import GaussianPrior
from orquestra.drug.discovery.docking.utils import process_molecule
from orquestra.drug.discovery.validator.filter_abstract import FilterAbstract
from torch.optim.lr_scheduler import CosineAnnealingLR

# from qumedl.models.priors import QCBMPrior
from orquestra.drug.discovery.validator import (
    GeneralFilter,
    PainFilter,
    WehiMCFilter,
    SybaFilter,
)
from orquestra.drug.metrics import MoleculeNovelty, get_diversity
from orquestra.drug.utils import ConditionFilters
import wandb  # Import wandb
import os
from datetime import datetime
import sys
import torch
import torch.nn as nn
import cloudpickle

## RBM
import optax
from orquestra.qml.models.rbm.jx import RBM
from orquestra.qml.api import Batch

# Initialize Qiskit Runtime Service with specific credentials
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


class RBMModel(RBM):
    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        random_seed=32,
        optimizer=optax.sgd(learning_rate=1e-6),
    ):
        super().__init__(
            n_visible, n_hidden, random_seed=random_seed, optimizer=optimizer
        )
        self.num_qubits = self.n_visible

    def train(self, data, probs, n_epoch):
        rbm_batch = Batch(data=data, probs=probs)
        # rbm_batch.batch_size = -1
        all_resuls = []
        for i in range(n_epoch):
            all_resuls.append(self._train_on_batch(rbm_batch))
        return all_resuls


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


project_dir_path, project_name, project_today = create_project_log_folder()

DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # needs to be cuda on the cluster
if len(sys.argv) > 2:

    prior_name = str(sys.argv[1])
    prior_size = int(sys.argv[2])
    random_seed = int(sys.argv[3])
    if DEVICE == "cuda":

        cuda_device_code = sys.argv[4]
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_code
    dataset_arg = str(sys.argv[5])
    if str(sys.argv[6]) == "sim":
        backend_sim = True
    else:
        backend_sim = False
    wandb_active = True

else:
    print("no input")
    prior_name = "rbm"
    prior_size = 16  # int(sys.argv[2])
    random_seed = 0
    cuda_device_code = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_code
    dataset_arg = "tartarus"
    backend_sim = True
    wandb_active = False
# os.environ["CUDA_VISIBLE_DEVICES"]
print(
    f"prior_name:{prior_name},prior_size:{prior_size},DEVICE:{DEVICE},cuda_device_code:{cuda_device_code},dataset_arg:{dataset_arg},random_seed:{random_seed}"
)

# DEVICE = 'cpu'
batch_size = 2028
prior_dim = prior_size

model_dim = embedding_dim = 256  # should be embedding_dim/n_attn_heads
n_attn_heads = 4
n_encoder_layers = 2

n_g_samples = 1000

dropout = 0.35912916665692707  # 0.2

n_epochs = 100
learning_rate = 0.0006225117554833231  # 1e-3
min_learning_rate = 1e-6
gradient_accumulation_steps = 5

n_epochs_prior = 30
n_test_samples = 1000

# dataset_name = "/home/mghazi/workspace/insilico-drug-discovery/data/KRAS_G12D/KRAS_G12D_inhibitors_451_modified.csv"
if dataset_arg == "tiny":

    dataset_name = "/root/qcbm/example/data/tiny.csv"

elif dataset_arg == "tartarus":
    dataset_name = "/root/generative-models/scripts/data/docking_hill_climbing_0.csv"
else:
    dataset_name = "/root/qcbm/example/data/full.csv"


pickle_name = dataset_name.split(".")[0]
if os.path.isfile(f"{pickle_name}.pkl"):
    selfies = load_object(f"{pickle_name}.pkl")
else:
    selfies = Selfies.from_smiles_csv(dataset_name)
    save_object(selfies, f"{pickle_name}.pkl")

smiles_dataset_df = pd.read_csv(dataset_name)
smiles_dataset = smiles_dataset_df.smiles.to_list()

selfies_dataset = selfies.as_dataset()

dl_shuffler = torch.Generator()
dl_shuffler.manual_seed(random_seed)


if prior_name == "random":
    prior = GaussianPrior(dim=prior_dim)
    prior_trainable = False
# elif prior_name == "qcbm":
#     entangling_layer_builder = LineEntanglingLayerBuilder(prior_dim)
#     ansatz = EntanglingLayerAnsatz(
#         prior_dim, n_qcbm_layers, entangling_layer_builder, use_rxx=False
#     )

# options = {
#     "maxiter": 5,  # Maximum number of iterations
#     "tol": 1e-6,  # Tolerance for termination
#     "disp": True,  # Display convergence messages
# }
# # Powell
# optimizer = ScipyOptimizer(method="COBYLA", options=options)

# prior = SingleBasisQCBM(ansatz, optimizer, distance_measure=ExactNLLTorch())

# prior_trainable = True
elif prior_name == "rbm":
    prior = RBMModel(
        n_visible=prior_dim,
        n_hidden=2 * prior_dim,
        random_seed=random_seed,
        optimizer=optax.sgd(learning_rate=1e-6),
    )
    prior_trainable = True
# Optional: Add configuration to wandb


# wandb.init(project=project_name, entity="mghazivakili")

run = wandb.init(
    # Set the project where this run will be logged
    # project=project_today,
    project="QPat-RBM",
    name=f"{project_name}-{prior_name}-{prior_dim}",
    # Track hyperparameters and run metadata
)

wandb.config = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "n_epochs": n_epochs,
    "model_dim": model_dim,
    "n_attn_heads": n_attn_heads,
    "n_encoder_layers": n_encoder_layers,
    "dropout": dropout,
    "prior_name": prior_name,
    "gpu_no": torch.cuda.device_count(),
    "project_dir_path": project_dir_path,
    "schduler": f"CosineAnnealingLR(optimizer, T_max={n_epochs}, eta_min={min_learning_rate})",
}


model = CausalMolPAT(
    vocab_size=selfies.n_tokens,
    embedding_dim=embedding_dim,
    prior_dim=prior.num_qubits,
    model_dim=model_dim,
    n_attn_heads=n_attn_heads,
    n_encoder_layers=n_encoder_layers,
    hidden_act=NewGELU(),
    dropout=dropout,
    padding_token_idx=selfies.pad_index,
)
# model = torch.compile(model)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # Wrap the model with nn.DataParallel
    model = nn.DataParallel(model)
    batch_size = batch_size * torch.cuda.device_count()


wandb.watch(model, log_freq=100)
# model = torch.compile(model)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=min_learning_rate)
print(DEVICE)


selfies_dl = DataLoader(
    selfies_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=dl_shuffler,
    collate_fn=TensorBatchCollator(),
)

# rewards
filter_lists = [TartarusFilters()]  # ,SybaFilter()]
weight_lists = [5.0]  # , 30.0]

novelity = MoleculeNovelty(smiles_dataset, threshold=0.6)
filter = ConditionFilters(filter_lists=filter_lists, weight_lists=weight_lists)

# get_diversity


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Non-trainable parameters
non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print(f"Trainable parameters: {trainable_params}")
print(f"Non-trainable parameters: {non_trainable_params}")
wandb.config.update(
    {"nt_parameter": non_trainable_params, "t_parameter": trainable_params}
)
# exit()
# training loop
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(total=len(selfies_dl), desc="Training Model") as prog_bar:
        prog_bar.set_description(f"Epoch {epoch} / {n_epochs}.")
        tensor_batch: TensorBatch
        for step, tensor_batch in enumerate(selfies_dl):
            tensor_batch.to(DEVICE)
            if prior_trainable:
                if prior_name == "rbm":
                    # torch.tensor(np.asarray(prior.generate(2,1))).to("cuda:0")
                    prior_samples = torch.tensor(
                        np.asarray(
                            prior.generate(
                                tensor_batch.batch_size, random_seed=random_seed
                            )
                        )
                    ).to(DEVICE)

            else:
                prior_samples = prior.generate(tensor_batch.batch_size).to(DEVICE)

            total_loss = compute_transformer_loss(
                model,
                tensor_batch,
                prior_samples=prior_samples,
            )

            total_loss.backward()

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()

            step_losses = {"total_loss": total_loss.item()}

            prog_bar.set_postfix(step_losses)
            prog_bar.update()

            tensor_batch.to(DEVICE)
            prior_samples.to(DEVICE)
        wandb.log({"total_loss": total_loss.item()})

    prog_bar.set_description("Generating test molecules")

    # generate a few samples and save them as JSON locally and to WandB
    model.eval()
    if prior_trainable:
        if prior_name == "rbm":
            if epoch >= 1:

                x_input = generated_before.cpu().numpy()
                # rbm_batch = Batch(data = x_input.astype(np.int64), probs=probs.cpu().numpy())
                # rbm_batch.batch_size = -1
                result = prior.train(
                    data=x_input.astype(np.int64), probs=probs.cpu().numpy(), n_epoch=20
                )
            test_prior_samples_before = torch.tensor(
                np.asarray(prior.generate(n_test_samples, random_seed=random_seed))
            ).to(DEVICE)
    start_tokens = torch.full(
        (n_test_samples, 1),
        fill_value=selfies.start_index,
        device=DEVICE,
        dtype=torch.int,
    )
    # generated_before_mol = model.generate(
    #     start_tokens, test_prior_samples_before, max_new_tokens=selfies.max_length
    # )
    if isinstance(model, torch.nn.DataParallel):
        generated_before_mol = model.module.generate(
            start_tokens, test_prior_samples_before, max_new_tokens=selfies.max_length
        )
    else:
        generated_before_mol = model.generate(
            start_tokens, test_prior_samples_before, max_new_tokens=selfies.max_length
        )
    test_molecules = selfies.decode(generated_before_mol.cpu().numpy())
    # print(test_molecules)

    ligands_before = selfies.selfie_to_smiles(test_molecules)
    novelity_rate = novelity.get_novelity_smiles(ligands_before)
    sr_rate_before = filter.get_validity_smiles(ligands_before)
    diversity_rate_before = get_diversity(ligands_before)

    rewards = []
    for lig in ligands_before:
        rewards.append(filter.compute_reward(lig)[1])
    soft = torch.nn.Softmax(dim=0)
    probs = soft(torch.Tensor(rewards))
    generated_before = test_prior_samples_before
    # print(probs)
    if prior_trainable:
        # Start a session
        if prior_name == "rbm":
            test_prior_samples_after = torch.tensor(
                np.asarray(prior.generate(n_test_samples, random_seed=random_seed))
            ).to(DEVICE)
        if isinstance(model, torch.nn.DataParallel):
            generated_after = model.module.generate(
                start_tokens,
                test_prior_samples_after,
                max_new_tokens=selfies.max_length,
            )
        else:
            generated_after = model.generate(
                start_tokens,
                test_prior_samples_after,
                max_new_tokens=selfies.max_length,
            )

        test_molecules_after = selfies.decode(generated_after.cpu().numpy())
        # print(test_molecules)

        ligands_after = selfies.selfie_to_smiles(test_molecules_after)
        sr_rate_after = filter.get_validity_smiles(ligands_before)
        diversity_rate_after = get_diversity(ligands_before)
        novelity_rate = novelity.get_novelity_smiles(ligands_before)
    sr_rate_before = filter.get_validity_smiles(ligands_before)
    print(
        f"sr_rate_before:{sr_rate_before},diversity_rate_before:{diversity_rate_before}"
    )  # ,novelity_rate:{novelity_rate_before}")
    if prior_trainable:
        print(
            f"sr_rate_after:{sr_rate_after},diversity_rate_after:{diversity_rate_after}"
        )
    else:
        diversity_rate_after = 0
        sr_rate_after = 0
    prog_bar.set_description(
        f"sr_rate_before:{sr_rate_before},sr_rate_after:{sr_rate_after},diversity_rate_before:{diversity_rate_before},,diversity_rate_after:{diversity_rate_after}"
    )

    wandb.log(
        {
            "sr_rate_before": sr_rate_before,
            "sr_rate_after": sr_rate_after,
            "novelity_rate_before": novelity_rate,
            "diversity_rate_before": diversity_rate_before,
            "diversity_rate_after": diversity_rate_after,
        }
    )

    # Optionally save your model at the end of each epoch or only at the end of training
    model_save_path = f"{project_dir_path}/model_epoch_{epoch}.pt"
    samples_epoch_save_path = f"{project_dir_path}/samples_epoch_{epoch}.csv"
    df = pd.DataFrame({"smiles": ligands_after})
    df.to_csv(samples_epoch_save_path)
    torch.save(model.state_dict(), model_save_path)

    wandb.save(model_save_path)
    wandb.save(samples_epoch_save_path)
    # UNCOMMENT to save samples as JSON
    # with open(f"test_molecules-{epoch}.json", "w") as f:
    #     json.dump(test_molecules, f)

save_obj(
    [model, selfies, novelity, selfies_dataset, ligands_after],
    f"{project_dir_path}/model_final.pkl",
)


# sample from the train model
if prior_trainable:
    if prior_name == "rbm":
        test_prior_samples = torch.tensor(
            np.asarray(prior.generate(n_g_samples, random_seed=random_seed))
        ).to(DEVICE)

else:
    test_prior_samples = prior.generate(n_g_samples).to(DEVICE)
start_tokens = torch.full(
    (n_g_samples, 1),
    fill_value=selfies.start_index,
    device=DEVICE,
    dtype=torch.int,
)

if isinstance(model, torch.nn.DataParallel):
    generated = model.module.generate(
        start_tokens, test_prior_samples, max_new_tokens=selfies.max_length
    )
else:
    generated = model.generate(
        start_tokens, test_prior_samples, max_new_tokens=selfies.max_length
    )


test_molecules = selfies.decode(generated.cpu().numpy())

ligands = selfies.selfie_to_smiles(test_molecules)
df = pd.DataFrame({"smiles": ligands})
final_path_save = f"{project_dir_path}/g_smiles.csv"
df.to_csv(final_path_save)
wandb.save(final_path_save)
# f"{project_dir_path}/model_final.pkl")
wandb.finish()
