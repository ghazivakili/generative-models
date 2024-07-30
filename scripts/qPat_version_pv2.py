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
# from qumedl.models.priors import QCBMPrior
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
from qcbm.qcbm_ibm import SingleBasisQCBM
from qcbm.circuit import LineEntanglingLayerBuilder,EntanglingLayerAnsatz
from qcbm.loss import ExactNLLTorch
from qcbm.optimizer import ScipyOptimizer
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit.primitives import Sampler
from qiskit_aer import Aer
## RBM
import optax
from orquestra.qml.models.rbm.jx import RBM
from orquestra.qml.api import Batch
# Initialize Qiskit Runtime Service with specific credentials
service = QiskitRuntimeService(name="ibm_uoft")
backend = service.backend("ibm_quebec")  # Using IBM Quebec backend


backend = Aer.get_backend('aer_simulator')

class RBMModel(RBM):
    def __init__(self,n_visible: int,n_hidden: int,random_seed=32,optimizer=optax.sgd(learning_rate=1e-6)):
        super().__init__(n_visible,n_hidden,random_seed=random_seed,optimizer=optimizer)
        self.num_qubits = self.n_visible

    def train(self,data,probs,n_epoch):
        rbm_batch = Batch(data = data, probs=probs)
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
    return (project_dir_path,project_name,project_today)

project_dir_path,project_name,project_today = create_project_log_folder()

DEVICE =  "cuda" if torch.cuda.is_available() else "cpu" # needs to be cuda on the cluster 
if len(sys.argv)>2:
    
    prior_name =  str(sys.argv[1]) 
    prior_size = int(sys.argv[2]) 
    random_seed = int(sys.argv[3])
    if DEVICE=="cuda":
        
        cuda_device_code=sys.argv[4]
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_code
    dataset_arg = str(sys.argv[5])
    if str(sys.argv[6]) == "sim":
        backend_sim = True
    else:
        backend_sim = False
    wandb_active = True
    
else:
    prior_name =  "rbm"
    prior_size = 8# int(sys.argv[2]) 
    random_seed = 0
    cuda_device_code = '7'
    os.environ["CUDA_VISIBLE_DEVICES"]=cuda_device_code
    dataset_arg = "tiny"
    backend_sim = True
    wandb_active = False

print(f"prior_name:{prior_name},prior_size:{prior_size},DEVICE:{DEVICE},cuda_device_code:{cuda_device_code},dataset_arg:{dataset_arg},random_seed:{random_seed}")

# DEVICE = 'cpu'
batch_size = 256
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
n_epochs_prior = 30 
n_test_samples = 500

# dataset_name = "/home/mghazi/workspace/insilico-drug-discovery/data/KRAS_G12D/KRAS_G12D_inhibitors_451_modified.csv"
if dataset_arg == "tiny":

    dataset_name = "/root/qcbm/example/data/tiny.csv"
else:
    dataset_name = "/root/qcbm/example/data/full.csv"
selfies = Selfies.from_smiles_csv(
  dataset_name  
)
smiles_dataset_df = pd.read_csv(dataset_name)
smiles_dataset = smiles_dataset_df.smiles.to_list()

selfies_dataset = selfies.as_dataset()

dl_shuffler = torch.Generator()
dl_shuffler.manual_seed(random_seed)


if backend_sim:
    # backend = service.backend("ibm_quebec")  # Using IBM Quebec backend


    backend = Aer.get_backend('aer_simulator')
    print("using Simulator")



if prior_name == "random":
    prior = GaussianPrior(dim=prior_dim)
    prior_trainable = False
elif prior_name == "qcbm":
    entangling_layer_builder = LineEntanglingLayerBuilder(prior_dim)
    ansatz = EntanglingLayerAnsatz(prior_dim, n_qcbm_layers, entangling_layer_builder,use_rxx=False)

    options = {
        'maxiter': 5,   # Maximum number of iterations
        'tol': 1e-6,      # Tolerance for termination
        'disp': True      # Display convergence messages
    }
    #Powell
    optimizer = ScipyOptimizer(method='COBYLA', options=options)

    prior = SingleBasisQCBM(ansatz, optimizer,distance_measure=ExactNLLTorch())

    prior_trainable = True
elif prior_name == "rbm":
    prior = RBMModel(
        n_visible=prior_dim,
        n_hidden=2*prior_dim,
        random_seed=random_seed,
        optimizer=optax.sgd(learning_rate=1e-6),
    )
    prior_trainable = True
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
    prior_dim=prior.num_qubits,
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
            if prior_trainable:
                if prior_name == "qcbm":
                    with Session(service=service, backend=backend,max_time=3600) as session:
                        sampler = Sampler()
                        sampler.set_options(
                            session=session,
                            resilience_level=2,
                            optimization_level=3,
                            error_mitigation={"method": "zne"},
                            shots = tensor_batch.batch_size
                        )                    
                        prior_samples = prior.generate(tensor_batch.batch_size, sampler, backend).to(DEVICE)
                elif prior_name=="rbm":
                    # torch.tensor(np.asarray(prior.generate(2,1))).to("cuda:0")
                    prior_samples =  torch.tensor(np.asarray(prior.generate(tensor_batch.batch_size,random_seed=random_seed))).to(DEVICE)
                    
            else:
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
    model.eval()
    if prior_trainable:
        # Start a session
        if prior_name == "qcbm":
            with Session(service=service, backend=backend,max_time=3600) as session:
                # sampler = Sampler(session=session,resilience_level =2, optimization_level=3,error_mitigation={"method": "zne"} )
                sampler = Sampler()
                sampler.set_options(
                    session=session,
                    resilience_level=2,
                    optimization_level=3,
                    error_mitigation={"method": "zne"},
                    shots=n_test_samples
                ) 
                if epoch >=1: 
                    x_input = generated_before.cpu().numpy()

                    result = prior.train_on_batch(x_input.astype(np.int64), probs.cpu().numpy(), sampler, backend, n_epochs_prior)
                test_prior_samples_before = prior.generate(n_test_samples, sampler, backend).to(DEVICE)
        elif prior_name == "rbm":
            if epoch >=1:
                
                x_input = generated_before.cpu().numpy()
                # rbm_batch = Batch(data = x_input.astype(np.int64), probs=probs.cpu().numpy())
                # rbm_batch.batch_size = -1
                result = prior.train(data = x_input.astype(np.int64), probs=probs.cpu().numpy(),n_epoch=20)
            test_prior_samples_before =  torch.tensor(np.asarray(prior.generate(n_test_samples,random_seed=random_seed))).to(DEVICE)
    start_tokens = torch.full(
        (n_test_samples, 1),
        fill_value=selfies.start_index,
        device=DEVICE,
        dtype=torch.int,
    )
    generated_before_mol = model.generate(
        start_tokens, test_prior_samples_before, max_new_tokens=selfies.max_length
    )
    test_molecules = selfies.decode(generated_before_mol.cpu().numpy())
    # print(test_molecules)
    
    ligands_before = selfies.selfie_to_smiles(test_molecules)
    # novelity_rate = novelity.get_novelity_smiles(ligands)
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
        if prior_name == "qcbm":
            with Session(service=service, backend=backend,max_time=3600) as session:
                # sampler = Sampler(session=session,resilience_level =2, optimization_level=3,error_mitigation={"method": "zne"} )
                sampler = Sampler()
                sampler.set_options(
                    session=session,
                    resilience_level=2,
                    optimization_level=3,
                    error_mitigation={"method": "zne"},
                    shots=n_test_samples
                ) 
                test_prior_samples_after = prior.generate(n_test_samples, sampler, backend).to(DEVICE)
                path_to_qcbm_params = f"{project_dir_path}/hw_param_train_{prior_dim}_qubits_{n_qcbm_layers}_layer_linear.json"
                prior.save_params(path_to_qcbm_params)
            # for sample, prob in zip(test_prior_samples_after, probabilities):
            #     print(f"Sample: {sample}, Probability: {prob}")
                wandb.save(path_to_qcbm_params)
        elif prior_name == "rbm":
            test_prior_samples_after =  torch.tensor(np.asarray(prior.generate(n_test_samples,random_seed=random_seed))).to(DEVICE)
            
        generated_after = model.generate(start_tokens, test_prior_samples_after, max_new_tokens=selfies.max_length)
        test_molecules_after = selfies.decode(generated_after.cpu().numpy())
        # print(test_molecules)
        
        ligands_after = selfies.selfie_to_smiles(test_molecules_after)
        sr_rate_after = filter.get_validity_smiles(ligands_before)
        diversity_rate_after = get_diversity(ligands_before)
        # novelity_rate = novelity.get_novelity_smiles(ligands)
    sr_rate_before = filter.get_validity_smiles(ligands_before)
    print(f"sr_rate_before:{sr_rate_before},diversity_rate_before:{diversity_rate_before}")#,novelity_rate:{novelity_rate_before}")
    if prior_trainable:
        print(f"sr_rate_after:{sr_rate_after},diversity_rate_after:{diversity_rate_after}")
    else:
        diversity_rate_after= 0
        sr_rate_after = 0
    prog_bar.set_description(f"sr_rate_before:{sr_rate_before},sr_rate_after:{sr_rate_after},diversity_rate_before:{diversity_rate_before},,diversity_rate_after:{diversity_rate_after}")
    
    
    wandb.log({"sr_rate_before":sr_rate_before,"sr_rate_after": sr_rate_after, "diversity_rate_before": diversity_rate_before, "diversity_rate_after": diversity_rate_after})

    # Optionally save your model at the end of each epoch or only at the end of training
    model_save_path = f"{project_dir_path}/model_epoch_{epoch}.pt"
    torch.save(model.state_dict(), model_save_path)
    
    wandb.save(model_save_path)
    # UNCOMMENT to save samples as JSON
    # with open(f"test_molecules-{epoch}.json", "w") as f:
    #     json.dump(test_molecules, f)
    
save_obj([model,selfies,novelity,selfies_dataset,ligands_after],f"{project_dir_path}/model_final.pkl")


# sample from the train model
if prior_trainable:
    if prior_name == "qcbm":
        with Session(service=service, backend=backend,max_time=3600) as session:
            # sampler = Sampler(session=session,resilience_level =2, optimization_level=3,error_mitigation={"method": "zne"} )
            sampler = Sampler()
            sampler.set_options(
                session=session,
                resilience_level=2,
                optimization_level=3,
                error_mitigation={"method": "zne"},
                shots=n_g_samples
            ) 
            test_prior_samples = prior.generate(n_g_samples, sampler, backend).to(DEVICE)
    elif prior_name == "rbm":
        test_prior_samples =  torch.tensor(np.asarray(prior.generate(n_g_samples,random_seed=random_seed))).to(DEVICE)
        
else:        
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
