import json
from pathlib import Path

import click
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from more_itertools import chunked
from torchmetrics.functional.classification import binary_f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..data.loaders import make_whole_peptide_set
from ..data.utils import seed_everything, split_peptide_set


def load_data(data_path: Path, tokenizer: AutoTokenizer, cv_split: int = 0, seed: int = 3245) -> DatasetDict:

    dataset = make_whole_peptide_set(data_path)
    dataset_dict = split_peptide_set(dataset, val=True, cv_split=cv_split, seed=seed)

    dataset_dict = DatasetDict({key : Dataset.from_pandas(df) for key, df in dataset_dict.items()})
    dataset_dict = dataset_dict.map(lambda sample : tokenizer(sample['peptide']))

    return dataset_dict

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    return {'f1': binary_f1_score(torch.Tensor(predictions), torch.Tensor(labels))}

def train(data_path: Path, output_path: Path, cv_split: int, model: str, seed: int):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2, problem_type='single_label_classification')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataset = load_data(data_path, tokenizer, cv_split=cv_split, seed=seed)

    training_args = TrainingArguments(
        output_dir=output_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=15,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    predict_dataset = dataset['test']
    predict_dataset = predict_dataset.map(lambda sample : tokenizer(sample['peptide'], return_tensors='pt', padding=True), batched=True, batch_size=16)
    device = model.device

    model_test_output = list()
    for batch in chunked(predict_dataset, n=16):
        
        inputs = {
                    "input_ids": torch.LongTensor([sample["input_ids"] for sample in batch]).to(device),
                    "attention_mask": torch.BoolTensor([
                        sample["attention_mask"] for sample in batch
                    ]).to(device),
                }
        
        with torch.no_grad():
            outputs = model.forward(**inputs)
        
        model_test_output.extend(outputs["logits"].cpu())

    model_test_output = torch.stack(model_test_output)
    predictions = torch.argmax(model_test_output, dim=-1)

    f1_score = binary_f1_score(predictions, torch.Tensor(predict_dataset['labels']))

    return f1_score


@click.command()
@click.option("--data_path", type=Path, required=True)
@click.option("--output_path", type=Path, required=True)
@click.option("--model", type=str, default='facebook/esm2_t12_35M_UR50D')
@click.option("--seed", type=int, default=3245)
def main(data_path: Path, output_path: Path, model: str, seed: int):

    seed_everything(seed)
    
    f1 = list()
    for cv_split in range(5):
        f1.append(train(data_path, output_path, cv_split, model, seed))

    with (output_path / 'results.json').open('w') as results_file:
        json.dump({'f1': {'mean': np.mean(f1).astype(float), 'std': np.std(f1).astype(float)}}, results_file)
    
