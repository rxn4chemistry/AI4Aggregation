from typing import Dict

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import random
import torch
import numpy as np

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


