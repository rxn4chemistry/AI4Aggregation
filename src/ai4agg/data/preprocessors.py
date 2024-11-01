import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ai4agg.util import FingerPrintCalculator

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

    def __init__(self, data: pd.DataFrame, random_state: int = 3245, **kwargs) -> None: # noqa
        super().__init__(data, random_state)

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
            n_pad = self.max_sequence_length - len(peptide)
            pad_vector = self.one_hot_encoder.transform([["pad"]])
            shaped_pad_vector = np.repeat(pad_vector, n_pad, 0)
            one_hot_sequence.append(shaped_pad_vector)

        one_hot_sequence = np.concatenate(one_hot_sequence, axis=0).flatten()

        return one_hot_sequence


@dataclass
class FingerprintPreprocessor(CorePreprocessor):

    def __init__(self, data: pd.DataFrame, n_fingerprint_bits: int = 128, random_state: int = 3245, **kwargs):
        super().__init__(data, random_state)

        self.fingerprint_calculator = FingerPrintCalculator(n_fingerprint_bits)
        self.maximum_sequence_length = 128 * self.maximum_sequence_length

    def __call__(self, peptide: str) -> np.ndarray:


        
    def process(
        self, data: pd.DataFrame, normalise: Optional[bool] = None
    ) -> Dict[str, Optional[np.ndarray]]:
        if normalise is None:
            normalise = self.normalise
        feature_data = self._peptides(data["peptide"])
        return feature_data

    def _peptides(self, peptides: pd.Series) -> np.ndarray:
        processed_peptides = list()

        if self.mode == "amino_acid":
            processed_peptides = self._process_amino_acid(peptides)
        elif self.mode == "whole_sequence":
            processed_peptides = process_map(
                self.fingerprint_calculator, peptides, max_workers=8, chunksize=10
            )

        return processed_peptides

    def _process_amino_acid(self, peptides: pd.Series):
        processed_peptides = list()
        for peptide in tqdm.tqdm(peptides):
            sequence_fingerprint = list()
            for amino_acid in list(peptide):
                fingerprint = self.fingerprint_calculator(amino_acid)
                sequence_fingerprint.append(fingerprint)

            sequence_fingerprint = np.hstack(sequence_fingerprint)

            # Pad in case of Amino Acids
            sequence_fingerprint = np.pad(
                sequence_fingerprint,
                ((0, self.maximum_sequence_length - sequence_fingerprint.shape[0])),
            )

            processed_peptides.append(sequence_fingerprint)

        return processed_peptides


@dataclass
class OccurencyVectorPreprocessor(CorePreprocessor):

    def __init__(self, data: pd.DataFrame, random_state: int = 3245, normalise: bool = True, **kwargs): # noqa
        super().__init__(data, random_state)

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
