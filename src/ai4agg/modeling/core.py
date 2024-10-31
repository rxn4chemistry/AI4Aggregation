from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def train_model_whole_peptide(data: pd.DataFrame, preprocessing_fn: Callable, model_class: Callable, metric_fn: Callable, seed: int = 3245):

    metrics = list()

    data['input'] = data['peptide'].map(preprocessing_fn)
    k_fold = KFold(5, shuffle=True, random_state=seed)
    split_indices = k_fold(data)

    for i in range(5):
        train_data, test_data = data.iloc[split_indices[i][0]], data.iloc[split_indices[i][1]]

        model = model_class()
        model.fit(train_data['input'].to_numpy(), train_data['aggregation'].to_numpy())

        pred = model.predict(test_data['input'].to_numpy())
        metric = metric_fn(test_data['aggregation'].to_numpy(), pred)
        metrics.append(metric)
    
    return np.mean(metrics), np.std(metrics)
