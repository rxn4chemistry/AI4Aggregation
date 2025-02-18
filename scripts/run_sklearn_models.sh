#!/bin/bash

top_folder=$1

for loader in reaction_set whole_set; do
    for preprocessor in sequence one_hot fingerprint occurency; do
        for model in rff xgb knn gaussian; do
            mkdir -p ${top_folder}/${loader}/${preprocessor}/${model}
            echo ${loader}/${preprocessor}/${model}

            poetry run train_sklearn_model \
                --data_path data/combined_data.csv\
                --output_path ${top_folder}/${loader}/${preprocessor}/${model} \
                --loader ${loader} \
                --preprocessor ${preprocessor} \
                --model ${model}
        done
    done
done


for model in hc2 timeforest weasel; do
    mkdir -p ${top_folder}/whole_set/sequence/${model}
    echo whole_set/sequence/${model}

    poetry run train_sklearn_model \
        --data_path data/combined_data.csv\
        --output_path ${top_folder}/whole_set/sequence/${model} \
        --loader whole_set \
        --preprocessor sequence \
        --model ${model}
done

