import random
from pathlib import Path

import numpy as np
import pandas as pd


# Reaction set: Data as is, one amino acid addition at a time
def make_reaction_set(data_path: Path, **kwargs) -> pd.DataFrame: # noqa
    data = pd.read_csv(data_path, index_col=0)
    data = data.reset_index(drop=True)
    return data

# Whole Peptide set: Only consider whole peptides
def make_whole_peptide_set(data_path: Path, **kwargs) -> pd.DataFrame: # noqa
    data = make_reaction_set(data_path)

    whole_peptide_indices = list()
    aggregation = list()
    for serial in data["serial"].unique():
        subset = data[data["serial"] == serial]
        whole_peptide_indices.append(subset.iloc[-1].name)
        aggregation.append(subset['aggregation'].sum() >= 1)

    whole_peptide_set = data.loc[whole_peptide_indices]
    whole_peptide_set['aggregation'] = aggregation
    whole_peptide_set = whole_peptide_set[["peptide", "serial", "aggregation"]]
    whole_peptide_set = whole_peptide_set.drop_duplicates(subset="peptide")

    return whole_peptide_set

# Whole Peptide Set shuffled: Consider whole peptides but shuffle them
def make_shuffled_peptide_set(data_path: Path, seed: int, **kwargs) -> pd.DataFrame: # noqa
    random.seed(seed)
    peptide_set = make_whole_peptide_set(data_path)
    peptide_set['peptide'] = peptide_set['peptide'].map(lambda peptide : ''.join(random.sample(list(peptide), len(peptide))))
    return peptide_set

# Shift point of aggregation
def get_aggregation_label_wof(
    subset: pd.DataFrame, start: int, end: int, drop: bool = False
):
    agg_points = np.argwhere(subset["aggregation"]).flatten()
    label = np.zeros(len(subset))
    if len(agg_points) == 0:
        subset["aggregation"] = label.astype(bool)
        return subset

    agg_point = agg_points[0]
    label[max(0, agg_point - start) : min(agg_point + end, len(subset))] = 1
    subset["aggregation"] = label.astype(bool)

    if drop:
        subset = subset.iloc[: min(agg_point + end, len(subset))]

    return subset[["peptide", "serial", "aggregation"]]

def make_wof_peptide_set(
    data_path: Path, wof_start: int, wof_end: int, wof_drop: bool = False, **kwargs # noqa
) -> pd.DataFrame:
    data = make_reaction_set(data_path)
    chunks = list()
    for serial in data["serial"].unique():
        subset = data[data["serial"] == serial]
        chunks.append(get_aggregation_label_wof(subset.copy(), wof_start, wof_end, wof_drop))

    return pd.concat(chunks)


# Aggregation Point: Drop amino acids after aggregation point
def make_agg_point_peptide_set(data_path: Path, **kwargs) -> pd.DataFrame: # noqa
    data = make_reaction_set(data_path)

    subset_indices = list()
    for serial in data["serial"].unique():
        subset = data[data["serial"] == serial]

        if subset['aggregation'].sum() == 0:
            subset_indices.append(subset.iloc[-1].name)
            continue
        
        agg_point = min(np.argwhere(subset['aggregation']).flatten())
        if agg_point <= 5:
            continue

        subset_indices.append(subset.iloc[agg_point].name)

    data = data.loc[subset_indices]
    return data

        
