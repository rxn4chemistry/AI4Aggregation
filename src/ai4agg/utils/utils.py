import random
from typing import Dict, List

import numpy as np
import pandas as pd
import pkg_resources
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import KFold, train_test_split

AA_TO_SMILES_PATH = pkg_resources.resource_filename(
    "ai4agg", "resources/onelet_to_smiles.csv"
)


def seed_everything(seed: int):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def split_peptide_set(dataset: pd.DataFrame, val: bool = False, cv_split: int = 0, seed: int = 3245) -> Dict[str, pd.DataFrame]:
    
    serials = dataset['serial'].unique()

    kfold = KFold(shuffle=True, random_state=seed)
    train_indices, test_indices = list(kfold.split(serials))[cv_split]
    train_serials, test_serials = serials[train_indices], serials[test_indices]

    train_set, test_set = dataset[dataset['serial'].isin(train_serials)], dataset[dataset['serial'].isin(test_serials)]

    dataset_dict = {'train': train_set, 'test': test_set}

    if val:
        train_set, val_set = train_test_split(dataset_dict['train'], test_size=0.1, random_state=seed)
        dataset_dict['train'] = train_set
        dataset_dict['val'] = val_set

    return dataset_dict



class FingerPrintCalculator:

    def __init__(self, n_bits: int) -> None:

        df = pd.read_csv(AA_TO_SMILES_PATH)
        df = df.replace(np.nan, "", regex=True)
        df["AASMILES"] = df["left"] + df["sidechain"] + df["right"]
        onelet_smiles_dict = dict(zip(df.abbrev, df.AASMILES))
        
        self.onelet_smiles_dict = onelet_smiles_dict
        self.n_bits = n_bits
        self.fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=n_bits)


    def smilifier(self, sequence: str) -> str:
        sequence_list = [*sequence][::-1]
        sequence_smiles_list = [
            self.onelet_smiles_dict.get(item, item) for item in sequence_list
        ]
        sequence_smiles_list += ["N"]
        sequence_smiles = "".join(sequence_smiles_list)
        return sequence_smiles

    def morgan_fingerprint(self, amino_acid: str) -> List[float]:
        smile = self.smilifier(amino_acid)
        mol = Chem.MolFromSmiles(smile)
        fp = self.fingerprint.GetFingerprintAsNumPy(mol)
        return list(fp)



