import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#from ai4agg.util import MordredCalculator, MorganCalculator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class Preprocessor:
    random_state: int = 3245

    def __call__(self, peptide: str) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SequencePreprocessor(Preprocessor):
    normalise: bool = field(default=True)

    normalisation_dict: dict = field(init=False)
    max_sequence_len: int = field(init=False)

    def initialise(
        self, data: pd.DataFrame
    ):
        self.max_sequence_len = max(data['peptide'].map(len))

    def __call__(self, peptide: str) -> np.ndarray:

        processed_peptide = np.zeros(self.max_sequence_len)
        for i, aa in enumerate(peptide):
            processed_peptide[i] = ord(aa)

        return processed_peptide


@dataclass
class OneHot_Sequence(Preprocessor):

    one_hot_encoder: OneHotEncoder = field(init=False)
    max_sequence_length: int = field(init=False)

    def initialise(self, data: pd.DataFrame) -> None:
        self.max_sequence_length = self.max_sequence_length = max(
            self.data["peptide"].map(len)
        )

        # Initialise OneHot Encoder
        peptides = self.data["peptide"].to_list()

        aa_frequencies = list()
        for peptide in peptides:
            amino_acids = list(peptide)
            amino_acids = [[aa] for aa in amino_acids]

            peptide_data = list()
            peptide_data.extend(amino_acids)
            peptide_data.append(["pad"])

            aa_frequencies.extend(peptide_data)

        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(aa_frequencies)

        self.one_hot_encoder = encoder


    def __call__(self, peptide: str,) -> np.ndarray:
        

        feature_data = self.one_hot_peptide(peptide)
        return feature_data

    def one_hot_peptide(self, peptides: pd.Series) -> np.ndarray:
        
        # One hot encoding
        one_hot_sequence = None
        for aa in peptides:
            one_hot_aa = self.one_hot_encoder.transform([[aa]])
            one_hot_aa = one_hot_aa.reshape(1, -1)

            if one_hot_sequence is None:
                one_hot_sequence = one_hot_aa
            else:
                one_hot_sequence = np.concatenate(
                    (one_hot_sequence, one_hot_aa), -1
                )

            # Padding
            n_pad = self.max_sequence_length - len(amino_acids)
            pad_vector = self.one_hot_encoder.transform([["pad"]])
            shaped_pad_vector = np.repeat(pad_vector, n_pad, 0)
            one_hot_sequence = np.concatenate(
                (one_hot_sequence[0], shaped_pad_vector.flatten())
            )

            x_peptides.append(one_hot_sequence)

        return x_peptides

"""
@dataclass
class Fingerprint_Sequence(Processor):
    normalise: bool = field(default=True)
    fingerprint: str = "mordred"
    mode: str = "amino_acid"
    maximum_sequence_length: int = 50
    regression: bool = False

    normalisation_dict: dict = field(init=False)
    fingerprint_calculator: Any = field(init=False)

    def __post_init__(
        self,
    ):
        if self.fingerprint == "mordred":
            self.fingerprint_calculator = MordredCalculator()
            self.maximum_sequence_length = 3692 * self.maximum_sequence_length
        elif self.fingerprint == "morgan":
            self.fingerprint_calculator = MorganCalculator()
            self.maximum_sequence_length = 2048 * self.maximum_sequence_length
        elif self.fingerprint == "rdkit":
            raise NotImplementedError
        else:
            raise ValueError

        if self.normalise:
            self.normalisation_dict = self.get_normalisations()

    def get_normalisations(self) -> Dict[str, Dict[str, float]]:
        normalisation_dict = dict()

        # Assume all features except 'peptides' are numerical and don't need special processing
        for feature in self.features_included:
            if feature == "peptide" and self.fingerprint != "mordred":
                continue
            elif feature == "peptide":
                relevant_feature = self._mordred_peptides(self.data[feature])
            else:
                relevant_feature = self.data[feature].to_numpy()

            mean = relevant_feature.mean(-1)
            std = relevant_feature.std(-1)
            normalisation_dict[feature] = {"mean": mean, "std": std}

        return normalisation_dict

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
class Occurency_Vector(Processor):
    normalise: bool = field(default=True)
    all_aa: list = field(init=False)

    def __post_init__(self):
        self.all_aa = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]

    def process(
        self, data: pd.DataFrame, normalise: Optional[bool] = None
    ) -> Dict[str, Optional[np.ndarray]]:
        if normalise is None:
            normalise = self.normalise

        x = list()
        for peptide in data["peptide"]:
            processed_peptide = self.get_occurency_vector(peptide)
            if self.normalise:
                processed_peptide = processed_peptide / len(peptide)
            x.append(processed_peptide)
        return x

    def get_occurency_vector(self, peptide):
        occ_vec = self.build_occurency_vector(peptide)
        return occ_vec

    def build_occurency_vector(self, peptide: str, max_len: int = 150):
        peptide = peptide[: min(len(peptide), max_len)]

        occurency_vector = np.zeros(len(self.all_aa))
        for i, aa in enumerate(self.all_aa):
            occurency_vector[i] = peptide.count(aa)
        return occurency_vector
"""
