import logging
from pathlib import Path
from typing import List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import panda as pd
import shap
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from ..utils.loaders import make_agg_point_peptide_set
from ..utils.preprocessors import OccurencyVectorPreprocessor
from ..utils.utils import seed_everything

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_motifs(peptide: str, motifs: str) -> List[str]:

    if motifs == '1_2':
        motifs = [peptide[i:i+1] for i in range(len(peptide) - 1)]
    elif motifs == '1_3':
        motifs = [peptide[i:i+2] for i in range(len(peptide) - 2)]
    else:
        raise ValueError(f"Unknown Motif type: {motifs}")

    motifs = [''.join(sorted(motif)) for motif in motifs]
    return motifs

def get_motif_occurency_vector(peptide: str, all_motifs: List[str]) -> np.ndarray:
    motif_occurency = np.zeros(len(all_motifs))

    for i, motif in enumerate(all_motifs):
        motif_occurency[i] = peptide.count(motif) + peptide.count(f"{motif[1]}{motif[0]}")
    
    return motif_occurency


def load_data(data_path: Path, motifs: str, normalise: bool, seed: int) -> Tuple[List[str], pd.DataFrame]:

    data = make_agg_point_peptide_set(data_path)

    preprocessor = OccurencyVectorPreprocessor(data, random_state=seed, normalise=normalise)
    data['input'] = data['peptide'].map(preprocessor)

    feature_names = preprocessor.all_aa
    if motifs in ['1_2', '1_3']:

        all_motifs = list()
        for peptide in data['peptide']:
            all_motifs.extend(get_motifs(peptide, motifs))

        motif_histogram = pd.value_counts(all_motifs)
        selected_motifs = motif_histogram[motif_histogram.values >= 20].index.to_list()

        data['input'] = data.apply(lambda row : np.concatenate([row['input'], get_motif_occurency_vector(row['peptide'], selected_motifs)]), axis=1)
        feature_names += selected_motifs

    return feature_names, data

def train(data: pd.DataFrame, n_repeats: int = 50):

    x_test = list()
    shap_values = list()
    f1_test = list()

    for _ in tqdm.tqdm(n_repeats):
        train_set, test_set = train_test_split(data, shuffle=True)

        model = XGBClassifier()
        model.fit(train_set['input'], train_set['aggregation'])

        pred_test = model.predict(test_set['input'].to_list())
        f1_test_seed = f1_score(test_set['aggregation'].to_list(), pred_test, average='micro')

        x_test_seed = np.stack(test_set['input'].to_list())
        explainer = shap.TreeExplainer(model)
        shap_values_seed = explainer.shap_values(x_test_seed)

        x_test.append(x_test_seed)
        shap_values.append(shap_values_seed)
        f1_test.append(f1_test_seed)

    f1_test = np.array(f1_test)
    x_test = np.concatenate(x_test)
    shap_values = np.concatenate(shap_values)

    return f1_test, x_test, shap_values


@click.command()
@click.option("--data_path", type=Path, required=True)
@click.option("--output_path", type=Path, required=True)
@click.option("--motifs", type=click.Choice(["no_motifs", "1_2", "1_3"]), default="no_motifs")
@click.option("--normalise", type=bool, default=True)
@click.option("--n_repeats", type=int, default=50)
@click.option("--seed", type=int, default=3245)
def main(data_path: Path, output_path: Path, motifs: str, normalise: bool, n_repeats: int, seed: int) -> None:

    seed_everything(seed)

    logger.info("Loading Data")
    feature_names, data = load_data(data_path, motifs, normalise, seed)

    logger.info("Training Models")
    f1_test, x_test, shap_values = train(data, n_repeats)
    logger.info(f"F1 Score: {np.mean(f1_test):.3f}±{np.std(f1_test):.3f}")

    logger.info("Explaining Predictions")
    plt.figure(dpi=300)
    shap.summary_plot(shap_values[:,:,1], x_test, feature_names=feature_names, plot_size=(13, 7.5), show=False)
    plt.savefig(output_path / 'explainer_results.png', dpi=300, bbox_inches='tight')

