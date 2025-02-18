# AI4Aggregation

This repo contains code accompanying the publication [Amino Acid Composition drives Peptide Aggregation: Predicting Aggregation for Improved Synthesis](https://chemrxiv.org/engage/chemrxiv/article-details/67a9af9ffa469535b9b67865), including scripts to reproduce all results.

<p align='center'>
  <img src='figure/Graphical Abstract.png' width="1000px">
</p>


## Installation

The project was developed and tested on Python 3.10 and utilise poetry as package manager. To install the package run the commands provided below. The installation is expected to take less than five minutes.  

```console
pip install poetry
poetry install
```

## Preprocessing data

To create the dataset used to train models, use the following script:

```console
poetry run create_combined_data --uzh_dataset_path <Path to UZH dataset> --mit_dataset_path <Path to MIT dataset> --save_path data/combined_data.csv
```

For this both the UZH and MIT dataset are required. For the UZH dataset download the file `uzh_data_clean.csv` from the [Zenodo record corresponding (version 1)](https://zenodo.org/records/14824562) to this publication. The MIT dataset can be found in the corresponding GitHub repo [here (last accessed 10.02.2025)](https://github.com/learningmatter-mit/peptimizer/blob/master/dataset/data_synthesis/synthesis_data.csv).

## Training models

### Classical Machine learning models

The following script allows training of classical machine learning models on the peptide aggregation data. This includes models ranging from a Random Forest to time series classification models:

```console
poetry run train_sklearn_model 
    --data_path: Path to UZH/MIT combined dataset
    --output_path: Path where the result of the training are going to be saved
    --loader: Data loader to represent the data. Choose from: [reaction_set (stepwise representation), whole_set (whole peptide) and whole_set_shuffled]
    --preprocessor: Preprocessor for the data. Choose from: [sequence, one_hot, fingerprint, occurency]
    --model: Model to train. Choose from: [rff (Random Forest), xgb (XGBoost), knn (K-Nearest Neighbour), gaussian (Gaussian Processes), hc2 (Hive Cote 2), timeforest (Time series forest), weasel (WEASEL)]
```





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
