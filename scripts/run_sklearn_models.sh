#!/bin/bash

top_folder=/dccstor/malberts_storage/UZH/ai4agg/experiments

for loader in reaction_set whole_set whole_set_shuffled; do
    for model in rff xgb knn gaussian; do
        mkdir -p ${top_folder}/sequence/${loader}/${model}
        echo sequence/${loader}/${model}

        poetry run train_sklearn_model \
             --data_path data/combined_data.csv\
             --output_path ${top_folder}/sequence/${loader}/${model} \
             --loader ${loader} \
             --preprocessor sequence \
             --model ${model}
    done
done