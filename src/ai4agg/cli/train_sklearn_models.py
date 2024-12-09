import json
import logging
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import click
import numpy as np
import pandas as pd
import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sktime.classification.dictionary_based import WEASEL
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import TimeSeriesForestClassifier
from xgboost import XGBClassifier

from ..utils.loaders import (
    make_reaction_set,
    make_shuffled_peptide_set,
    make_whole_peptide_set,
    make_wof_peptide_set,
)
from ..utils.preprocessors import (
    FingerprintPreprocessor,
    OccurencyVectorPreprocessor,
    OneHotPreprocessor,
    SequencePreprocessor,
)
from ..utils.utils import seed_everything, split_peptide_set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


LOADER_REGISTRY = {'reaction_set': make_reaction_set,
                   'wof_set': make_wof_peptide_set,
                   'whole_set': make_whole_peptide_set,
                   'whole_set_shuffled': make_shuffled_peptide_set}

PREPROCESSOR_REGISTRY = {'sequence': SequencePreprocessor,
                         'one_hot': OneHotPreprocessor,
                         'fingerprint': FingerprintPreprocessor,
                         'occurency': OccurencyVectorPreprocessor
                         }

MODEL_REGISTRY = {'rff': RandomForestClassifier,
                  'xgb': XGBClassifier,
                  'knn': KNeighborsClassifier,
                  'gaussian': GaussianProcessClassifier,
                  'hc2': partial(HIVECOTEV2, time_limit_in_minutes=10),
                  'timeforest': TimeSeriesForestClassifier,
                  'weasel': WEASEL
                  }


def load_data(data_path: Path, loader: str, preprocessor: str, cv_split: int = 0, seed: int = 3245, **kwargs) -> pd.DataFrame:

    dataset = LOADER_REGISTRY[loader](data_path, seed=seed, **kwargs) # type: ignore

    preprocessor = PREPROCESSOR_REGISTRY[preprocessor](dataset, random_state=seed, **kwargs)
    dataset['input'] = dataset['peptide'].map(preprocessor)

    if loader == 'whole_set_shuffled':
        train_set, test_set = train_test_split(dataset, random_state=seed)
        dataset_dict = {'train': train_set, 'test': test_set}
    else:
        dataset_dict = split_peptide_set(dataset, val=False, cv_split=cv_split, seed=seed)

    return dataset_dict

def train(dataset_dict: Dict[str, pd.DataFrame], model: str) -> float:

    classifier = MODEL_REGISTRY[model]()
    classifier.fit(np.stack(dataset_dict['train']['input'].to_list()), np.stack(dataset_dict['train']['aggregation'].to_list()))

    test_set = dataset_dict['test']
    test_set['prediction'] = classifier.predict(np.stack(test_set['input'].to_list()))

    ground_truth = list()
    predictions = list()
    for serial in test_set['serial'].unique():
        subset = test_set[test_set['serial'] == serial]

        ground_truth.append(np.any(subset['aggregation']))
        predictions.append(np.any(subset['prediction']))
    
    f1 = f1_score(ground_truth, predictions)
    accuracy = accuracy_score(ground_truth, predictions)
    return f1, accuracy
    

@click.command()
@click.option("--data_path", type=Path, required=True)
@click.option("--output_path", type=Path, required=True)
@click.option("--loader", type=click.Choice(list(LOADER_REGISTRY.keys())))
@click.option("--preprocessor", type=click.Choice(list(PREPROCESSOR_REGISTRY.keys())))
@click.option("--model", type=click.Choice(list(MODEL_REGISTRY.keys())))
@click.option("--seed", type=int, default=3245)
@click.option("--n_repeats", type=int, default=0)
@click.option("--wof_start", type=int, required=False)
@click.option("--wof_end", type=int, required=False)
@click.option("--wof_drop", type=bool, required=False)
@click.option("--occurency_vector_normalise", type=bool, required=False)
def main(data_path: Path,
         output_path: Path,
         loader: str,
         preprocessor: str,
         model: str,
         seed: int,
         n_repeats: int,
         wof_start: Optional[int],
         wof_end: Optional[int],
         wof_drop: Optional[bool],
         occurency_vector_normalise: Optional[bool],
         ) -> None:

    seed_everything(seed)
    
    f1, accuracy = list(), list()

    if loader == 'whole_set_shuffled':
        for i in tqdm.tqdm(range(n_repeats)):
            dataset_dict = load_data(data_path,
                                     loader,
                                     preprocessor,
                                     cv_split=0,
                                     padding=True,
                                     wof_start=wof_start,
                                     wof_end=wof_end,
                                     wof_drop=wof_drop,
                                     occurency_vector_normalise=occurency_vector_normalise,
                                     seed=i)
            
            f1_result, accuracy_results = train(dataset_dict, model)
            f1.append(f1_result)
            accuracy.append(accuracy_results)
    
    else:
        for cv_split in range(5):
            logger.info(f"Running Split {cv_split}/5")

            if model in ['hc2', 'timeforest', 'weasel'] and (loader != 'whole_set' or preprocessor != 'sequence'):
                raise ValueError(f"Incompatible loader {loader} or preprocessor {preprocessor} for model {model}.")

            dataset_dict = load_data(data_path,
                                     loader,
                                     preprocessor,
                                     cv_split=cv_split,
                                     padding=True,
                                     wof_start=wof_start,
                                     wof_end=wof_end,
                                     wof_drop=wof_drop,
                                     occurency_vector_normalise=occurency_vector_normalise,
                                     seed=seed)
            
            f1_result, accuracy_results = train(dataset_dict, model)
            logger.info(f"Finished Running Split {cv_split}/5 F1: {f1_result:.3f}")

            f1.append(f1_result)
            accuracy.append(accuracy_results)

    logger.info(f"F1: {np.mean(f1):.3f}±{np.std(f1):.3f}")
        
    with (output_path / 'results.json').open('w') as results_file:
        json.dump({'f1': {'mean': np.mean(f1).astype(float), 'std': np.std(f1).astype(float)}, 'raw_f1': list(f1), 'raw_acc': list(accuracy)}, results_file)
    
