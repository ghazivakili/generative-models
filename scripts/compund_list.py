from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd

# Load the data
df = pd.read_csv("/root/generative-models/scripts/qcbm_list.csv")

# Create a list of RDKit molecule objects from the SMILES strings
molecules = [Chem.MolFromSmiles(smiles) for smiles in df["SMILES"]]

# Create a grid of images for the molecules
img = Draw.MolsToGridImage(
    molecules,
    molsPerRow=4,
    subImgSize=(400, 400),
    legends=df["Molecule Name/ID"].tolist(),
)

# Save or show the image
img.save("/root/generative-models/scripts/molecules.png")  # Save as PNG
img.show()  # Show the image
