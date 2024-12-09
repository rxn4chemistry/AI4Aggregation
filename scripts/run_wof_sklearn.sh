#!/bin/bash

top_folder=/dccstor/malberts_storage/UZH/ai4agg/experiments/wof

for preprocessor in sequence one_hot fingerprint; do
    for model in rff xgb knn gaussian; do
        for wof_start in 0 2 4 6 8 10; do
            for wof_end in 0 2 4 6 8 10; do
                for wof_drop in True False; do

                    mkdir -p ${top_folder}/${preprocessor}/${model}/drop_${wof_drop}/start_${wof_start}/end_${wof_end}
                    echo ${top_folder}/${preprocessor}/${model}/drop_${wof_drop}/start_${wof_start}/end_${wof_end}

                    poetry run train_sklearn_model \
                        --data_path data/combined_data.csv\
                        --output_path ${top_folder}/${preprocessor}/${model}/drop_${wof_drop}/start_${wof_start}/end_${wof_end} \
                        --loader wof_set \
                        --preprocessor ${preprocessor} \
                        --model ${model} \
                        --wof_start ${wof_start} \
                        --wof_end ${wof_end} \
                        --wof_drop ${wof_drop}

                done
            done
        done
    done
done