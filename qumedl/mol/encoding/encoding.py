import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, MutableMapping, Optional, Tuple

import numpy as np
from numpy import typing as npt
import torch
from torch.utils.data import TensorDataset


class MolecularEncoding(ABC):
    def __init__(
        self,
        mol_strings: List[str],
        start_token: str,
        pad_token: str,
        max_length: Optional[int] = None,
        _index_token_mapping: MutableMapping[int, str] | None = None,
        _token_index_mapping: MutableMapping[str, int] | None = None,
    ) -> None:
        """Abstract base class for molecular encodings.

        Args:
            mol_strings (List[str]): List of molecular strings in SMILES format.
            start_token (str): start of sequence token.
            pad_token (str): padding token.
            max_length (Optional[int], optional): Maximum length of a molecular sequence. Defaults to None.
                If this value is None, the maximum length will be set to the length of the longest
                molecular string in ``mol_strings`` multiplied by ``fallback_max_length_factor``.
        """
        super().__init__()

        self.index_to_token = _index_token_mapping
        self.token_to_index = _token_index_mapping

        self.start_token = start_token
        self.pad_token = pad_token

        if self.index_to_token is None or self.token_to_index is None:
            self.token_to_index, self.index_to_token = self._build_vocab(
                mol_strings, start_token, pad_token
            )

        self.n_tokens = len(self.token_to_index)
        self.mol_strings = mol_strings

        self._max_length = (
            max_length if max_length is not None else len(max(mol_strings, key=len))
        )

    @classmethod
    def from_smi(
        cls,
        filename: str,
        start_token: str,
        pad_token: str,
        max_length: Optional[int] = None,
        file_encoding: str = "utf-8",
    ) -> "MolecularEncoding":
        """Reads molecule strings from a .smi file.

        Args:
            filename (str): Path to the .smi file.
            start_token (str): Token signifying the start of a molecular sequence.
            pad_token (str): Token signifying padding.
            max_length (int, optional): Maximum length of a molecular sequence. Defaults to None.
            file_encoding (str): Encoding of the .smi file. Defaults to "utf-8".

        Returns:
            MolecularEncoding: Instance of the MolecularEncoder class with data from the specified ".smi" file.
        """
        path_to_file = Path(filename).resolve()

        with open(path_to_file, encoding=file_encoding) as file:
            molecules_uncleaned = file.readlines()

        molecules_cleaned = [mol_str.strip() for mol_str in molecules_uncleaned]
        return cls(molecules_cleaned, start_token, pad_token, max_length)

    @classmethod
    def from_csv(
        cls,
        filename: str,
        column_name: str,
        start_token: str,
        pad_token: str,
        max_length: Optional[int] = None,
        file_encoding: str = "utf-8",
    ) -> "MolecularEncoding":
        """Reads molecule strings from a .csv file.

        Args:
            filename (str): Path to the .smi file.
            start_token (str): Token signifying the start of a molecular sequence.
            pad_token (str): Token signifying padding.
            max_length (int, optional): Maximum length of a molecular sequence. Defaults to None.
            file_encoding (str): Encoding of the .smi file. Defaults to "utf-8".

        Returns:
            MolecularEncoding: Instance of the MolecularEncoder class with data from the specified ".csv" file.
        """
        path_to_file = Path(filename).resolve()

        with open(path_to_file, encoding=file_encoding) as file:
            reader = csv.reader(file)
            molecule_idx = next(reader).index(column_name)
            data = [row[molecule_idx] for row in reader]

        return cls(data, start_token, pad_token, max_length)

    @abstractmethod
    def _build_vocab(
        self, mol_strings: List[str], start_token: str, pad_token: str
    ) -> Tuple[MutableMapping[str, int], MutableMapping[int, str]]:
        raise NotImplementedError

    @abstractmethod
    def _mol_string_len(self, mol_string: str) -> int:
        raise NotImplementedError

    def pad(self, mol_string: str, final_length: Optional[int] = None):
        """Adds the padding token specified during construction to a molecular string until it is of a specified length

        Args:
            mol_string (str): The mol string to pad.
            final_length (int): Final length of the padded string. Defaults to None.
                If the value is None, the final length will be set to the length of the longest
                molecular string in ``mol_strings`` multiplied by ``fallback_max_length_factor``.
                If this value is less than or equal to the length of ``mol_string`` then the
                molecular string will be returned unmodified.
        """
        n_missing = final_length - self._mol_string_len(mol_string)
        if n_missing <= 0:
            return mol_string

        return mol_string + self.pad_token * n_missing

    def unpad(self, mol_string: str) -> str:
        """Removes all instances of the padding token from a molecular string.

        Args:
            mol_string (str): Molecular string to remove padding from.

        Returns:
            str: unpadded molecular string.
        """
        return mol_string.rstrip(self.pad_token)

    def unpad_encoded(self, encoded_molecule: np.ndarray) -> np.ndarray:
        """Removes all instances of the padding token from an encoded molecular string.

        Args:
            encoded_molecule (np.ndarray): Encoded molecular string to remove padding from.

        Returns:
            np.ndarray: unpadded encoded molecular string.
        """
        return encoded_molecule[encoded_molecule != self.pad_index]

    @abstractmethod
    def _encode(self, mol_strings: List[str]) -> npt.NDArray[np.int32]:
        raise NotImplementedError

    def encode(self, mol_strings: List[str]) -> npt.NDArray[np.int32]:
        """Encodes a list of molecule strings.

        Args:
            mol_strings (List[str]): List of molecule strings.

        Returns:
            npt.NDArray[np.int32]: array of shape ``(len(mol_strings), max_length)``
        """
        return self._encode(mol_strings)

    def __array__(self, dtype: np.dtype = "int") -> npt.NDArray[np.int32]:
        return self.encode(self.mol_strings).astype(dtype)

    def asarray(
        self,
        dtype: np.dtype = "int32",
        include_start_token: bool = False,
    ) -> npt.NDArray[np.int32]:
        """Returns the encoded molecules as a numpy array.

        Args:
            dtype (np.dtype, optional): Data type of the returned array. Defaults to "int32".
            include_start_token (bool, optional): Whether to include the start token in the returned array. Defaults to True.
                Can be useful when using the returned array as input to a model.

        Returns:
            npt.NDArray[np.int32]: array of shape ``(len(mol_strings), max_length)``
        """
        if include_start_token:
            return np.concatenate(
                [
                    np.full((len(self.mol_strings), 1), self.start_index),
                    self.encode(self.mol_strings),
                ],
                axis=1,
            ).astype(dtype)

        return self.encode(self.mol_strings).astype(dtype)

    def start_array(
        self, n_molecules: int, dtype: np.dtype = "int32"
    ) -> npt.NDArray[np.int32]:
        """Returns an array of start tokens of shape (n_molecules, 1)

        Args:
            n_molecules (int): Number of molecules.
            dtype (np.dtype, optional): Data type of the returned array. Defaults to "int32".

        Returns:
            npt.NDArray[np.int32]: array of shape ``(n_molecules, 1)``
        """
        return np.full((n_molecules, 1), self.start_index).astype(dtype)

    def as_dataset(self, include_start_token: bool = True) -> TensorDataset:
        """Returns the encoded molecules as a TensorDataset.

        Returns:
            TensorDataset: TensorDataset of encoded molecules.
        """
        return TensorDataset(
            torch.tensor(self.asarray(include_start_token=include_start_token))
        )

    def as_tensor(
        self, dtype: np.dtype = "int32", include_start_token: bool = False
    ) -> torch.Tensor:
        """Returns the encoded molecules as a torch.Tensor.

        Args:
            dtype (np.dtype, optional): Data type of the returned array. Defaults to "int32".
            include_start_token (bool, optional): Whether to include the start token in the returned array. Defaults to True.
                Can be useful when using the returned array as input to a model.

        Returns:
            torch.Tensor: torch.Tensor of encoded molecules.
        """
        if include_start_token:
            return torch.cat(
                [
                    torch.full((len(self.mol_strings), 1), self.start_index),
                    torch.tensor(
                        self.asarray(include_start_token=False)
                    ),  # start token included in line above
                ],
                axis=1,
            ).astype(dtype)

        return torch.tensor(self.asarray(include_start_token=False))

    @abstractmethod
    def _decode(
        self, encoded_molecules: npt.NDArray[np.int32], stop_token_id: int
    ) -> List[str]:
        raise NotImplementedError

    def decode(
        self, encoded_molecules: npt.NDArray[np.int32], stop_token_id: int | None = None
    ) -> List[str]:
        """Decodes a tensor of encoded molecules and returns a list
        of decoded molecule strings.

        Args:
            encoded_molecules (npt.NDArray[np.int32]): tensor of encoded molecules.
            stop_token_id (int, optional): a token which if encountered indicates the end of the sequence.
                All tokens after will be removed from the encoded sequence. If unspecified then the,
                padding token will be used.

        Returns:
            List[str]: list of decoded molecule strings.
        """
        stop_token_id = stop_token_id or self.pad_index
        return self._decode(encoded_molecules, stop_token_id)

    def index_of(self, token: str) -> int:
        """Returns the index of a specified token.

        Args:
            token (str): token to get the index of.

        Returns:
            int: index of the token.
        """

        return self.token_to_index[token]

    @property
    def pad_index(self) -> int:
        """Returns the index of the padding token.

        Returns:
            int: index of the padding token.
        """
        return self.index_of(self.pad_token)

    @property
    def start_index(self) -> int:
        """Returns the index of the start token.

        Returns:
            int: index of the start token.
        """
        return self.index_of(self.start_token)

    @property
    def max_length(self) -> int:
        """Returns the maximum length of a molecular sequence.

        Returns:
            int: maximum length of a molecular sequence.
        """
        return self._max_length

    @max_length.setter
    def max_length(self, value: int) -> None:
        """Sets the maximum length of a molecular sequence.

        Args:
            value (int): maximum length of a molecular sequence.
        """
        self._max_length = value
