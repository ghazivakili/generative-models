from typing import List, MutableMapping, Tuple

import cloudpickle as pickle
import numpy as np
import pandas as pd
import selfies as sf
from numpy import typing as npt
from tqdm import tqdm

from .encoding import MolecularEncoding


class Selfies(MolecularEncoding):
    def _build_vocab(
        self, mol_strings: List[str], start_token: str, pad_token: str
    ) -> Tuple[MutableMapping[str, int], MutableMapping[int, str]]:
        vocab: set = sf.get_alphabet_from_selfies(mol_strings)
        vocab.add(pad_token)
        vocab.add(start_token)

        token_to_index = dict((token, index) for index, token in enumerate(vocab))
        index_to_char = {index: token for token, index in token_to_index.items()}

        return token_to_index, index_to_char

    @classmethod
    def from_smiles_csv(
        cls,
        filepath: str,
        start_token: str = "[^]",
        pad_token: str = "[nop]",
        column: str = "smiles",
        strict_encoder: bool = False,
        max_length: int | None = None,
    ) -> "Selfies":
        """Creates a Selfies instance from a csv file containing SMILES strings.

        Args:
            filepath (str): path to csv file.
            start_token (str, optional): start of sequence token. Defaults to "[^]".
            pad_token (str, optional): padding token. Defaults to "[nop]".
            column (str, optional): column name of SMILES strings in CSV file. Defaults to "smiles".
            strict_encoder (bool, optional): whether to use the strict encoder. Defaults to False.
            max_length (Optional[int], optional): Maximum length of a molecular sequence. Defaults to None.
                If this value is None, the maximum length will be set to the length of the longest
                molecular string in ``mol_strings`` multiplied by ``fallback_max_length_factor``.
            fallback_max_length_factor (float, optional): Factor to multiply the length of the longest
                molecular string in ``mol_strings`` by to get the maximum length. Defaults to 1.1.

        Returns:
            Selfies: Selfies instance.
        """
        df = pd.read_csv(filepath)
        smiles_mol_strings = df[column].tolist()

        selfies: list[str] = []
        for smile_mol_string in tqdm(
            smiles_mol_strings, desc="Loading Selfies From CSV"
        ):
            if "." in smile_mol_string:
                len_0 = len(smile_mol_string.split(".")[0])
                len_1 = len(smile_mol_string.split(".")[1])

                if len_0 > len_1:
                    smile_mol_string = smile_mol_string.split(".")[0]
                elif len_0 <= len_1:
                    smile_mol_string = smile_mol_string.split(".")[1]

            selfie_mol_string = sf.encoder(smile_mol_string, strict=strict_encoder)

            if selfie_mol_string is not None:
                selfies.append(selfie_mol_string)

        return cls(selfies, start_token, pad_token, max_length=max_length)

    @classmethod
    def from_pickle(cls, filepath: str) -> "Selfies":
        """Loads selifes from a pickle file.
        When loaded pickle file must be a dictionary with the following keys:
        "selfies": list[str]
        "start_token": str
        "pad_token": str
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        selfies = data["selfies"]
        start_token = data["start_token"]
        pad_token = data["pad_token"]

        return cls(selfies, start_token, pad_token)

    def save_pickle(self, filepath: str) -> None:
        """Saves selfies as pickle."""
        data = {
            "selfies": self.mol_strings,
            "start_token": self.start_token,
            "pad_token": self.pad_token,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def _mol_string_len(self, mol_string: str) -> int:
        """Returns the length of a molecular sequence.

        Args:
            mol_string (str): A molecular sequence.

        Returns:
            int: Length of the molecular sequence.
        """
        return sf.len_selfies(mol_string)

    def _encode(self, mol_strings: List[str]) -> npt.NDArray[np.int32]:
        encoded = np.empty((len(mol_strings), self.max_length))
        for mol_string_idx, mol_string in enumerate(mol_strings):
            padded_selfies = self.pad(mol_string, self.max_length)
            encoded_selfies = sf.selfies_to_encoding(
                padded_selfies, self.token_to_index, enc_type="label"
            )
            encoded[mol_string_idx] = encoded_selfies

        return encoded

    def _decode(
        self, encoded_molecules: npt.NDArray[np.int32], stop_token_id: int
    ) -> List[str]:
        decoded_sf_list = list()
        for encoded_sf in encoded_molecules.tolist():
            encoded_sf: list

            # drops everything that follows a stop token
            try:
                stop_token_position = encoded_sf.index(stop_token_id)
            except ValueError as val_error:
                stop_token_position = -1

            encoded_sf = encoded_sf[:stop_token_position]

            decoded_sf = sf.encoding_to_selfies(
                encoded_sf, self.index_to_token, enc_type="label"
            )

            # TODO: check with Mohamad if it matters doing this after decoding or ok to slice from
            # encoded molecules array
            if self.start_token in decoded_sf:
                decoded_sf = decoded_sf.replace(self.start_token, "")

            decoded_sf_list.append(decoded_sf)

        return decoded_sf_list

    def train_test_split(
        self,
        train_ratio: float | int = 0.8,
        shuffle: bool = True,
        random_seed: int | None = None,
    ) -> Tuple["Selfies", "Selfies"]:
        """Splits the data into two MolecularEncoding objects.

        Args:
            split_ratio (float or int): Ratio of data to be in the first split. Defaults to 0.8.
                If ``split_ratio`` is a float, it must be between 0 and 1. If ``split_ratio`` is an int,
                it will be interpreted as an absolute number of samples.
            shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
            random_seed (int, optional): Seed for the random number generator. Defaults to None.

        Returns:
            Tuple[Selfies, Selfies]: Tuple of two MolecularEncoding objects.
        """

        if shuffle:
            rng = np.random.default_rng(random_seed)
            rng.shuffle(self.mol_strings)

        if isinstance(train_ratio, float) and not 0 <= train_ratio <= 1:
            raise ValueError(
                f"train_ratio must be a float between 0 and 1, got {train_ratio}"
            )

        if isinstance(train_ratio, int) and train_ratio <= 0:
            raise ValueError(
                f"train_ratio must be an int greater than 0, got {train_ratio}"
            )

        n_train = (
            int(len(self.mol_strings) * train_ratio)
            if isinstance(train_ratio, float)
            else train_ratio
        )

        train_mol_strings = self.mol_strings[:n_train]
        test_mol_strings = self.mol_strings[n_train:]

        train_encoding = Selfies(
            train_mol_strings,
            self.start_token,
            self.pad_token,
            max_length=self._max_length,
            # keep the same mappings
            _index_token_mapping=self.index_to_token,
            _token_index_mapping=self.token_to_index,
        )

        test_encoding = Selfies(
            test_mol_strings,
            self.start_token,
            self.pad_token,
            max_length=self._max_length,
            _index_token_mapping=self.index_to_token,
            _token_index_mapping=self.token_to_index,
        )

        return train_encoding, test_encoding
    
    def selfie_to_smiles(self,selfie_list:List[str]):
        """sumary_line
        
        Keyword arguments:
        argument -- description
        Return: returns smiles
        """
        smiles = []
        for smi in selfie_list:
            smiles.append(sf.decoder(smi))
        
        return smiles
