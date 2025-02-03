#!/bin/bash

top_folder=$1

for model in facebook/esm2_t6_8M_UR50D facebook/esm2_t12_35M_UR50D facebook/esm2_t30_150M_UR50D facebook/esm2_t33_650M_UR50D google-bert/bert-base-uncased google-bert/bert-large-uncased; do
    mkdir -p ${top_folder}/${model}_pretrained
    echo ${top_folder}/${model}_pretrained

    poetry run train_hf_model \
            --data_path data/combined_data.csv\
            --output_path ${top_folder}/${model}_pretrained \
            --model ${model} \
            --pretrained True
done


for model in facebook/esm2_t6_8M_UR50D facebook/esm2_t12_35M_UR50D facebook/esm2_t30_150M_UR50D facebook/esm2_t33_650M_UR50D google-bert/bert-base-uncased google-bert/bert-large-uncased; do
    mkdir -p ${top_folder}/${model}
    echo ${top_folder}/${model}

    poetry run train_hf_model \
            --data_path data/combined_data.csv\
            --output_path ${top_folder}/${model} \
            --model ${model} \
            --pretrained False
done