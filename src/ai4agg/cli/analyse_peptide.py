import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import shap
import tqdm
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ..utils.preprocessors import OccurencyVectorPreprocessor
from ..utils.utils import seed_everything
from .explainability import get_motif_occurency_vector_1_2, load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def train(data: pd.DataFrame, n_repeats: int = 2):

    models = list()
    for _ in tqdm.tqdm(range(n_repeats)):
        train_set, _ = train_test_split(data, shuffle=True)

        model = XGBClassifier()
        model.fit(np.stack(train_set['input'].to_list()), np.stack(train_set['label'].to_list()))
        models.append(model)

    return models


@click.command()
@click.option("--train_data", type=Path, required=True)
@click.option("--peptide", type=str, required=True)
@click.option("--n_repeats", type=int, default=50)
@click.option("--seed", type=int, default=3245)
def main(train_data: Path, peptide: str, n_repeats: int, seed: int) -> None:

    seed_everything(seed)

    logger.info("Loading Data")
    feature_names, data = load_data(train_data, '1_2', True, seed)
    
    logger.info("Training Models")
    models = train(data, n_repeats)

    # Flip peptide and restrict to 20 aa
    peptide = ''.join(list(reversed(peptide)))[:20]

    preprocessor = OccurencyVectorPreprocessor(data, random_state=seed, normalise=True)

    occurency_vector = np.concatenate([preprocessor(peptide), get_motif_occurency_vector_1_2(peptide, feature_names[20:])])
    occurency_vector = np.expand_dims(occurency_vector, 0)

    agg = 0
    shap_values = list()
    for model in models:
        out = model.predict(occurency_vector)
        agg += 1 if (out.flatten()[1] == 1) else 0
        explainer = shap.TreeExplainer(model)
        shap_values.append(explainer.shap_values(occurency_vector)[:, :, 1])

    print(f"Sequence to analyse: {peptide}")
    print(f"Model Predictions: {agg}/{len(models)}")

    shap_values = np.mean(np.concatenate(shap_values), axis=0)

    relevant_section_peptide = peptide[1:min(len(peptide), 10)]

    print(f"Section with potential replacement: {relevant_section_peptide}")

    if 'S' not in relevant_section_peptide and 'T' not in relevant_section_peptide:
        print("No S or T found to replace.")
    else:
        motifs_in_peptide = get_motif_occurency_vector_1_2(relevant_section_peptide, feature_names[20:])
        relevant_motifs = list(filter(lambda motif: 'S' in motif or 'T' in motif, set(feature_names[20:]).intersection(motifs_in_peptide)))

        if 'S' in relevant_section_peptide and 'T' in relevant_section_peptide:
            impact_threonine = shap_values[feature_names.index('T')]
            impact_serine = shap_values[feature_names.index('S')]
            
            if impact_threonine > impact_serine:
                print('Threonine has higher impact -> Replace first.')
            else:
                print('Serine has higher impact -> Replace first.')

        elif 'S' in relevant_section_peptide:
            print('Replace Serine')
        elif 'T' in relevant_section_peptide:
            print('Replace Threonine')

        impact_list = list()
        for motif in relevant_motifs:
            impact_motif = shap_values[feature_names.index(motif)]
            if impact_motif > 0:
                if ('S' in motif and motif[0] != 'S') or ('T' in motif and motif[0] != 'T'):
                    motif = ''.join(reversed(motif))
                    if motif not in relevant_section_peptide:
                        continue
                    
                impact_list.append({'motif': motif, 'impact': impact_motif})
        

        if len(impact_list) == 0:
            print("No motifs to replace found.")
        else:
            impact_df = pd.DataFrame.from_dict(impact_list)
            impact_df = impact_df.sort_values(by='impact', ascending=False)
            for i in range(len(impact_df)):
                print(f"Motif: {impact_df.iloc[i]['motif']} Impact: {impact_df.iloc[i]['impact']:.3f}")

