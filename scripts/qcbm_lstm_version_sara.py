import json

import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import tqdm
import numpy as np
from qumedl.mol.encoding.selfies_ import Selfies
from qumedl.models.transformer.pat import CausalMolPAT
from qumedl.models.transformer.loss_functions import compute_transformer_loss
from qumedl.training.collator import TensorBatchCollator
from qumedl.training.tensor_batch import TensorBatch
from qumedl.models.activations import NewGELU
from qumedl.models.priors import GaussianPrior
from qumedl.models.priors import QCBMPrior
from orquestra.drug.discovery.validator import GeneralFilter, PainFilter, WehiMCFilter,SybaFilter
from orquestra.drug.metrics import MoleculeNovelty, get_diversity
from orquestra.drug.utils import  ConditionFilters
import wandb  # Import wandb
import os
from datetime import datetime
import sys
import torch
import torch.nn as nn
import cloudpickle

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
    return (project_dir_path,project_name,project_today)

project_dir_path,project_name,project_today = create_project_log_folder()



prior_name =  str(sys.argv[1]) 
prior_size = int(sys.argv[2]) 

random_seed = 42
DEVICE =  "cuda" if torch.cuda.is_available() else "cpu" # needs to be cuda on the cluster 
# DEVICE = 'cpu'
batch_size = 32
prior_dim = prior_size

model_dim = embedding_dim = 256 # should be embedding_dim/n_attn_heads
n_attn_heads = 16
n_encoder_layers = 4

n_g_samples = 500

dropout = 0.2

n_epochs = 30
learning_rate = 0.001
gradient_accumulation_steps = 4

n_qcbm_layers=4

n_test_samples = 500

# dataset_name = "/home/mghazi/workspace/insilico-drug-discovery/data/KRAS_G12D/KRAS_G12D_inhibitors_451_modified.csv"
dataset_name = "/home/mghazi/workspace/insilico-drug-discovery/data/merged_dataset/1Mstoned_vsc_initial_dataset_insilico_chemistry42_filtered.csv"

selfies = Selfies.from_smiles_csv(
  dataset_name  
)
smiles_dataset_df = pd.read_csv(dataset_name)
smiles_dataset = smiles_dataset_df.smiles.to_list()

selfies_dataset = selfies.as_dataset()

dl_shuffler = torch.Generator()
dl_shuffler.manual_seed(random_seed)






prior_guassian = GaussianPrior(dim=prior_dim)

prior_qcbm = QCBMPrior(dim=prior_dim, n_layer= n_qcbm_layers)


if prior_name == "guassian":
    prior = prior_guassian
elif prior_name == "qcbm":
    prior = prior_qcbm

# Optional: Add configuration to wandb


# wandb.init(project=project_name, entity="mghazivakili")
run = wandb.init(
    # Set the project where this run will be logged
    project=project_today,
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
  "prior_name":prior_name,
  "gpu_no":torch.cuda.device_count(),
  "project_dir_path":project_dir_path
}



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

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # Wrap the model with nn.DataParallel
    model = nn.DataParallel(model)
    batch_size = batch_size * torch.cuda.device_count()
    
    
wandb.watch(model, log_freq=100)

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(DEVICE)


selfies_dl = DataLoader(
    selfies_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=dl_shuffler,
    collate_fn=TensorBatchCollator(),
)

# rewards 
filter_lists=[GeneralFilter(), PainFilter(),WehiMCFilter()]#,SybaFilter()]
weight_lists=[15.0, 5.0, 5.0]#, 30.0]

novelity = MoleculeNovelty(smiles_dataset,threshold=0.6)
filter = ConditionFilters(filter_lists=filter_lists,weight_lists=weight_lists)
# get_diversity

# training loop
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(total=len(selfies_dl), desc="Training Model") as prog_bar:
        prog_bar.set_description(f"Epoch {epoch} / {n_epochs}.")
        tensor_batch: TensorBatch
        for step, tensor_batch in enumerate(selfies_dl):
            tensor_batch.to(DEVICE)
            prior_samples = prior.generate(tensor_batch.batch_size).to(DEVICE)
            total_loss = compute_transformer_loss(
                model, tensor_batch, prior_samples=prior_samples
            )

            total_loss.backward()

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            step_losses = {"total_loss": total_loss.item()}

            prog_bar.set_postfix(step_losses)
            prog_bar.update()

            tensor_batch.to(DEVICE)
            prior_samples.to(DEVICE)
        wandb.log({"total_loss": total_loss.item()})

    prog_bar.set_description("Generating test molecules")
    
    # generate a few samples and save them as JSON locally and to WandB
    test_prior_samples = prior.generate(n_test_samples).to(DEVICE)
    start_tokens = torch.full(
        (n_test_samples, 1),
        fill_value=selfies.start_index,
        device=DEVICE,
        dtype=torch.int,
    )
    
    model.eval()
    generated = model.module.generate(
        start_tokens, test_prior_samples, max_new_tokens=selfies.max_length
    )
    test_molecules = selfies.decode(generated.cpu().numpy())
    # print(test_molecules)
    
    ligands = selfies.selfie_to_smiles(test_molecules)
    novelity_rate = novelity.get_novelity_smiles(ligands)
    sr_rate = filter.get_validity_smiles(ligands)
    diversity_rate = get_diversity(ligands)
    
    
    rewards = []
    for lig in ligands:
        rewards.append(filter.compute_reward(lig)[1])
    soft = torch.nn.Softmax(dim=0)
    probs = soft(torch.Tensor(rewards))
    print(probs)
    print(f"sr_rate:{sr_rate},diversity_rate:{diversity_rate},novelity_rate:{novelity_rate}")
    prog_bar.set_description(f"sr_rate:{sr_rate},diversity_rate:{diversity_rate},novelity_rate:{novelity_rate}")
    
    
    wandb.log({"sr_rate": sr_rate, "diversity_rate": diversity_rate, "novelity_rate": novelity_rate})

    # Optionally save your model at the end of each epoch or only at the end of training
    model_save_path = f"{project_dir_path}/model_epoch_{epoch}.pt"
    torch.save(model.state_dict(), model_save_path)
    
    wandb.save(model_save_path)
    # UNCOMMENT to save samples as JSON
    # with open(f"test_molecules-{epoch}.json", "w") as f:
    #     json.dump(test_molecules, f)
save_obj([model,selfies,novelity,selfies_dataset,ligands],f"{project_dir_path}/model_final.pkl")


# sample from the train model
test_prior_samples = prior.generate(n_g_samples).to(DEVICE)
start_tokens = torch.full(
    (n_g_samples, 1),
    fill_value=selfies.start_index,
    device=DEVICE,
    dtype=torch.int,
)
generated = model.module.generate(
    start_tokens, test_prior_samples, max_new_tokens=selfies.max_length
)
test_molecules = selfies.decode(generated.cpu().numpy())

ligands = selfies.selfie_to_smiles(test_molecules)
df = pd.DataFrame({"smiles":ligands})
df.to_csv(f"{project_dir_path}/g_smiles.csv")
# f"{project_dir_path}/model_final.pkl")
wandb.finish()
