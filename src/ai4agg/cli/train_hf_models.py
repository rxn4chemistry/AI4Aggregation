import json
from pathlib import Path

import click
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from more_itertools import chunked
from torchmetrics.functional.classification import binary_accuracy, binary_f1_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..utils.loaders import make_whole_peptide_set
from ..utils.utils import seed_everything, split_peptide_set


def load_data(data_path: Path, tokenizer: AutoTokenizer, cv_split: int = 0, seed: int = 3245) -> DatasetDict:

    dataset = make_whole_peptide_set(data_path)
    dataset['labels'] = dataset['aggregation'].astype(int)
    dataset_dict = split_peptide_set(dataset, val=True, cv_split=cv_split, seed=seed)

    dataset_dict_hf = DatasetDict({key : Dataset.from_pandas(df) for key, df in dataset_dict.items()})
    dataset_dict_hf = dataset_dict_hf.map(lambda sample : tokenizer(sample['peptide']))

    return dataset_dict_hf

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    return {'f1': binary_f1_score(torch.Tensor(predictions), torch.Tensor(labels))}

def train(data_path: Path, output_path: Path, cv_split: int, model_name: str, pretrained: bool, seed: int):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, problem_type='single_label_classification')
    else:
        model_config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_config(model_config)
    
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

    model_test_output_tensor = torch.stack(model_test_output)
    predictions = torch.argmax(model_test_output_tensor, dim=-1)

    f1_score = binary_f1_score(predictions, torch.Tensor(predict_dataset['labels']))
    accuracy_score = binary_accuracy(predictions, torch.Tensor(predict_dataset['labels']))

    return f1_score.item(), accuracy_score.item()


@click.command()
@click.option("--data_path", type=Path, required=True)
@click.option("--output_path", type=Path, required=True)
@click.option("--model", type=str, default='facebook/esm2_t12_35M_UR50D')
@click.option("--pretrained", type=bool, default=True)
@click.option("--seed", type=int, default=3245)
def main(data_path: Path, output_path: Path, model: str, pretrained: bool, seed: int):

    seed_everything(seed)
    
    f1, accuracy = list(), list()
    for cv_split in range(5):
        f1_result, accuracy_result = train(data_path, output_path, cv_split, model, pretrained, seed)
        f1.append(f1_result)
        accuracy.append(accuracy_result)


    with (output_path / 'results.json').open('w') as results_file:
        json.dump({'f1': {'mean': np.mean(f1).astype(float), 'std': np.std(f1).astype(float)}, 'raw_f1': list(f1), 'raw_acc': list(accuracy)}, results_file)
    
