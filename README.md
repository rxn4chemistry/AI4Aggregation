# AI4Aggregation

This repo contains code accompanying the publication ["Amino Acid Composition drives Peptide Aggregation: Predicting Aggregation for Improved Synthesis"](to be added), including scripts to reproduce all results.

## Installation

This project utilises poetry as package manager. Install the package run:

```console
poetry install
```

## Preprocessing data

To create the dataset used to train models, use the following script:

```console
poetry run create_combined_data --uzh_dataset_path <Path to UZH dataset> --mit_dataset_path <Path to MIT dataset> --save_path data/combined_data.csv
```

For this both the UZH and MIT dataset are required. For the UZH dataset download the file `uzh_data_clean.csv` from the [Zenodo record corresponding](https://zenodo.org/records/14824562) to this publication. The MIT dataset can be found in the corresponding GitHub repo [here](https://github.com/learningmatter-mit/peptimizer/blob/master/dataset/data_synthesis/synthesis_data.csv). 

## Reproducing Training results

We provide a set of scripts to replicate the results obtained in the paper. All scripts require a path to where the results of the experiments are saved. The HuggingFace models require GPUs to train:
```console
bash scripts/run_hf_models.sh <Path to Experiment Folder>
bash scripts/run_sklearn_models.sh <Path to Experiment Folder>
bash scripts/run_sklearn_shuffled.sh <Path to Experiment Folder>
bash scripts/run_wof_sklearn.sh <Path to Experiment Folder>
```

## Explainability

To explain the predictions of the models we use Shap values. To reproduce our results use the following scripts:

```console
poetry run explain_model --data_path data/combined_data.csv --output_path <Path where results should be stored>
```
