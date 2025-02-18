# AI4Aggregation

This repo contains code accompanying the publication [Amino Acid Composition drives Peptide Aggregation: Predicting Aggregation for Improved Synthesis](https://chemrxiv.org/engage/chemrxiv/article-details/67a9af9ffa469535b9b67865), including scripts to reproduce all results.

<p align='center'>
  <img src='figure/GraphicalAbstract.png' width="1000px">
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

As an example run the following. The folder `results/xgb` needs to exist:

```console
poetry run train_sklearn_model --data_path data/combined_data.csv --output_path results/xgb  --loader whole_set --preprocessor occurency --model xgb
```

The expected accuracy of the model is `0.596±0.019`.

### Training HuggingFace models

To train HuggingFace models on the aggregation data use the following script. A GPU is recommended to train the models but at the expense of time the models can also be run locally:

```console
poetry run train_hf_model 
    --data_path: Path to UZH/MIT combined dataset
    --output_path: Path where the result of the training are going to be saved
    --model: Model to train e.g. facebook/esm2_t33_650M_UR50D. We evaluated ESM 2.0 and BERT.
    --pretrained: Wether to train the model from scratch or use the pretrained one from HuggingFace. [True False]
```


## Reproducing Training results

We provide a set of scripts to replicate the results obtained in the paper. All scripts require a path to where the results of the experiments are saved. A GPU is recommended to reproduce the Transformer based results:
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
