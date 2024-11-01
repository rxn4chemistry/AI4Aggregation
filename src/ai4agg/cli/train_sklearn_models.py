import json
import logging
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from ..utils.loaders import (
    make_reaction_set,
    make_shuffled_peptide_set,
    make_whole_peptide_set,
    make_wof_peptide_set,
)
from ..utils.preprocessors import SequencePreprocessor
from ..utils.utils import seed_everything, split_peptide_set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


LOADER_REGISTRY = {'reaction_set': make_reaction_set,
                   'wof_set': make_wof_peptide_set,
                   'whole_set': make_whole_peptide_set,
                   'whole_set_shuffled': make_shuffled_peptide_set}

PREPROCESSOR_REGISTRY = {'sequence': SequencePreprocessor,
                         'one_hot': None,
                         'fingerprint': None,
                         }

MODEL_REGISTRY = {'rff': RandomForestClassifier,
                  'xgb': XGBClassifier,
                  'knn': KNeighborsClassifier,
                  'gaussian': GaussianProcessClassifier,
                  'hc2': None,
                  'timeforest': None,
                  'weasel': None
                  }


def load_data(data_path: Path, loader: str, preprocessor: str, cv_split: int = 0, seed: int = 3245, **kwargs) -> pd.DataFrame:

    dataset = LOADER_REGISTRY[loader](data_path, **kwargs)

    preprocessor = PREPROCESSOR_REGISTRY[preprocessor](dataset, random_state=seed, **kwargs)
    dataset['input'] = dataset['peptide'].map(preprocessor)

    dataset_dict = split_peptide_set(dataset, val=False, cv_split=cv_split, seed=seed)

    return dataset_dict

def train(dataset_dict: Path, model: str, seed: int = 3245) -> float:

    classifier = MODEL_REGISTRY[model](random_state=seed)
    classifier.fit(dataset_dict['train']['input'].to_list(), dataset_dict['train']['aggregation'].to_list())

    test_set = dataset_dict['test']
    test_set['prediction'] = classifier.predict(test_set['input'].to_list())

    ground_truth = list()
    predictions = list()
    for serial in test_set['serial'].unique():
        subset = test_set[test_set['serial'] == serial]

        ground_truth.append(np.any(subset['aggregation']))
        predictions.append(np.any(subset['prediction']))
    
    f1 = f1_score(ground_truth, predictions)
    return f1
    

@click.command()
@click.option("--data_path", type=Path, required=True)
@click.option("--output_path", type=Path, required=True)
@click.option("--loader", type=click.Choice(LOADER_REGISTRY.keys()))
@click.option("--preprocessor", type=click.Choice(PREPROCESSOR_REGISTRY.keys()))
@click.option("--model", type=click.Choice(MODEL_REGISTRY.keys()))
@click.option("--seed", type=int, default=3245)
@click.option("--wof_start", type=int, required=False)
@click.option("--wof_end", type=int, required=False)
@click.option("--wof_drop", type=bool, required=False)
def main(data_path: Path,
         output_path: Path,
         loader: str,
         preprocessor: str,
         model: str,
         seed: int,
         wof_start: Optional[int],
         wof_end: Optional[int],
         wof_drop: Optional[bool],
         ) -> None:

    seed_everything(seed)
    
    f1 = list()
    for cv_split in range(5):
        logger.info(f"Running Split {cv_split}/5")

        dataset_dict = load_data(data_path, loader, preprocessor, cv_split=cv_split, wof_start=wof_start, wof_end=wof_end, wof_drop=wof_drop)
        f1_result = train(dataset_dict, model, seed)
        logger.info(f"Finished Running Split {cv_split}/5 F1: {f1_result:.3f}")

        f1.append(f1_result)
        
    with (output_path / 'results.json').open('w') as results_file:
        json.dump({'f1': {'mean': np.mean(f1).astype(float), 'std': np.std(f1).astype(float)}}, results_file)
    
