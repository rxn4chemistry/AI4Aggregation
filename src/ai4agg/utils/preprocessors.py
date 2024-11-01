import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .utils import FingerPrintCalculator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CorePreprocessor:
    
    def __init__(self, data: pd.DataFrame, padding: bool = True, random_state: int = 3245, **kwargs) -> None: # noqa

        self.padding = padding
        self.random_state = random_state
        self.max_sequence_len = max(data['peptide'].map(len))

    def __call__(self, peptide: str) -> np.ndarray: # noqa
        raise NotImplementedError


class SequencePreprocessor(CorePreprocessor):

    def __call__(self, peptide: str) -> np.ndarray:
        
        processed_peptide = np.zeros(self.max_sequence_len if self.padding else len(peptide))
        for i, aa in enumerate(peptide):
            processed_peptide[i] = ord(aa)

        return processed_peptide


class OneHotPreprocessor(CorePreprocessor):

    def __init__(self, data: pd.DataFrame, padding: bool = True, random_state: int = 3245, **kwargs) -> None: # noqa
        super().__init__(data, padding, random_state)

        # Initialise OneHot Encoder
        aa_frequencies = list()
        for peptide in data["peptide"].to_list():
            amino_acids = list(peptide)
            amino_acids = [[aa] for aa in amino_acids]

            aa_frequencies.extend(amino_acids + [["pad"]])

        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(aa_frequencies)

        self.one_hot_encoder = encoder

    def __call__(self, peptide: str,) -> np.ndarray:
        
        feature_data = self.one_hot_peptide(peptide)
        return feature_data

    def one_hot_peptide(self, peptide: str) -> np.ndarray:
        
        # One hot encoding
        one_hot_sequence = list()
        for aa in peptide:
            one_hot_aa = self.one_hot_encoder.transform([[aa]])
            one_hot_aa = one_hot_aa.reshape(1, -1)
            one_hot_sequence.append(one_hot_aa)

        # Padding
        if self.padding:
            n_pad = self.max_sequence_len - len(peptide)
            pad_vector = self.one_hot_encoder.transform([["pad"]])
            shaped_pad_vector = np.repeat(pad_vector, n_pad, 0)
            one_hot_sequence.append(shaped_pad_vector)

        one_hot_sequence_np = np.concatenate(one_hot_sequence, axis=0).flatten()

        return one_hot_sequence_np


@dataclass
class FingerprintPreprocessor(CorePreprocessor):

    def __init__(self, data: pd.DataFrame, padding: bool = True, n_fingerprint_bits: int = 128, random_state: int = 3245, **kwargs): # noqa
        super().__init__(data, padding, random_state)

        self.fingerprint_calculator = FingerPrintCalculator(n_fingerprint_bits)
        self.maximum_sequence_length = n_fingerprint_bits * self.max_sequence_len

    def __call__(self, peptide: str) -> np.ndarray:
        return np.zeros(len(peptide))


@dataclass
class OccurencyVectorPreprocessor(CorePreprocessor):

    def __init__(self, data: pd.DataFrame, random_state: int = 3245, normalise: bool = True, **kwargs): # noqa
        super().__init__(data, False, random_state) # Occurency Vector needs no padding

        self.all_aa = pd.unique(data['peptide'].map(lambda peptide : list(peptide)).to_list())
        self.normalise = normalise

    def __call__(self, peptide: str) -> np.ndarray:

        occurcency_vector = self.build_occurency_vector(peptide)
        if self.normalise:
            occurcency_vector = occurcency_vector / len(peptide)
        return occurcency_vector

    def build_occurency_vector(self, peptide: str):
        occurency_vector = np.zeros(len(self.all_aa))
        for i, aa in enumerate(self.all_aa):
            occurency_vector[i] = peptide.count(aa)
        return occurency_vector
