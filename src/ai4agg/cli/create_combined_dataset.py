from pathlib import Path

import click
import pandas as pd


def filter_synthesis(synthesis_data: pd.DataFrame) -> pd.DataFrame:
    subsets = list()
    for serial in synthesis_data['serial'].unique():
        subset = synthesis_data[synthesis_data['serial'] == serial]

        # Filter out all synthesis not starting from scratch
        if len(subset) != (len(subset['peptide'].iloc[-1]) - 1):
            continue

        # Filter out all short peptides
        if len(subset['peptide'].iloc[-1]) <= 5:
            continue

        # Shorten peptides to the first 20
        subset = subset[:min(len(subset), 20)]
        subsets.append(subset[['serial', 'peptide', 'amino_acid', 'first_diff']])

    return pd.concat(subsets)


def process_uzh_data(uzh_data: pd.DataFrame) -> pd.DataFrame:
    uzh_data['peptide'] = uzh_data["pre-chain"] + uzh_data["amino_acid"]
    uzh_data['serial'] = "uzh_" + uzh_data['serial'].astype(int).astype(str)

    uzh_data = filter_synthesis(uzh_data)
    return uzh_data


def process_mit_data(mit_data: pd.DataFrame) -> pd.DataFrame:
    mit_data['peptide'] = mit_data["pre-chain"] + mit_data["amino_acid"]
    mit_data['serial'] = "mit_" + mit_data['serial'].astype(str)

    mit_data = filter_synthesis(mit_data)
    return mit_data


def clean_cys_his(synthesis_data: pd.DataFrame, row: object) -> float:
    if not (row["amino_acid"] == "H" or row["amino_acid"] == "C"):
        return row["first_diff"]
    else:
        first_index = synthesis_data.iloc[0].name
        last_index = synthesis_data.iloc[-1].name
        # Cys His is the second amino acid -> Interpolate between 0 and the second
        if row.name == first_index:
            interpolated_diff = synthesis_data.loc[row.name + 1]["first_diff"] / 2
        # Cys His is the last amino acid -> Use the previous diff
        elif row.name == last_index:
            interpolated_diff = synthesis_data.loc[row.name - 1]["first_diff"]
        # Interpolate
        else:
            interpolated_diff = (
                synthesis_data.loc[row.name - 1]["first_diff"]
                + synthesis_data.loc[row.name + 1]["first_diff"]
            ) / 2

        return interpolated_diff


def get_clean_unique_peptides(peptide_df: pd.DataFrame):

    unique_peptides = list()
    unique_synth_sets = list()
    for serial in peptide_df['serial'].unique():
        subset = peptide_df[peptide_df['serial'] == serial]
        
        if subset['peptide'].iloc[-1] in unique_peptides:
            continue

        unique_peptides.append(subset['peptide'].iloc[-1])
        unique_synth_sets.append(subset)
    
    return pd.concat(unique_synth_sets)


@click.command()
@click.option("--uzh_dataset_path", type=Path, required=True)
@click.option("--mit_dataset_path", type=Path, required=True)
@click.option("--save_path", type=Path, required=True)
def main(uzh_dataset_path: Path, mit_dataset_path: Path, save_path: Path):

    # Process UZH Data
    uzh_data = pd.read_csv(uzh_dataset_path, index_col=0)
    processed_uzh_data = process_uzh_data(uzh_data)

    # Process MIT Data
    mit_data = pd.read_csv(mit_dataset_path)
    processed_mit_data = process_mit_data(mit_data)

    # Combine datasets
    combined_dataset = pd.concat([processed_uzh_data, processed_mit_data])

    # Clean Cysteine and Histidine peaks (Temperature change causes artifacts)
    combined_dataset["first_diff_clean"] = combined_dataset.apply(
        lambda row: clean_cys_his(
            combined_dataset[combined_dataset["serial"] == row["serial"]], row
        ),
        axis=1,
    )

    # Drop Duplicates
    combined_dataset = get_clean_unique_peptides(combined_dataset)

    # Add aggregation
    combined_dataset['aggregation'] = (combined_dataset['first_diff_clean'] < -0.2)

    # Save
    combined_dataset = combined_dataset[['serial', 'peptide', 'first_diff_clean', 'aggregation']]
    combined_dataset.to_csv(save_path)
