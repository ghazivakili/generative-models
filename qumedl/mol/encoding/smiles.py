from typing import List, MutableMapping, Optional, Tuple

import numpy as np
from numpy import typing as npt

from .encoding import MolecularEncoding


class Smiles(MolecularEncoding):
    """Class for manipulating SMILES strings.

    Attributes:
        index_to_token (MutableMapping[int, str]): Dictionary mapping indices to tokens.
        token_to_index (MutableMapping[str, int]): Dictionary mapping tokens to indices.
        padding_token (str): Token signifying padding.
        start_token (str): Token signifying the start of a molecular sequence.

    Methods:
        from_smi (@classmethod): Reads molecule strings from a .smi file.
        from_csv (@classmethod): Reads molecule strings from a .csv file.
        encode: Encodes a list of SMILES strings into a numpy array of integers.
        decode: Decodes a tensor of integers into a list of SMILES strings.
        pad: Pads a SMILES string to a specified length.
        unpad: Removes padding from a SMILES string.
    """

    def __init__(
        self,
        mol_strings: List[str],
        start_token: str = "^",
        pad_token: str = "_",
        max_length: Optional[int] = None,
    ) -> None:
        """Initialize a SmilesEncoder instance.

        Args:
            mol_strings (List[str]): list of SMILES strings.
            start_token (str): token indicating start of a SMILE sequence. Defaults to "^".
            pad_token (str): token indicating padding in a SMILE sequence. Defaults to "_".
            max_length (Optional[int], optional): _description_. Defaults to None.
        """
        super().__init__(mol_strings, start_token, pad_token, max_length)

    def _build_vocab(
        self, mol_strings: List[str], start_token: str, pad_token: str
    ) -> Tuple[MutableMapping[str, int], MutableMapping[int, str]]:
        """Builds the vocabulary of SMILES from a list of SMILES strings.

        Args:
            mol_strings (List[str]): List of SMILES strings.
            start_token (str): token signifying the start of a
                molecular sequence.
            pad_token (str): token signifying padding

        Returns:
            Tuple[MutableMapping[str, int], MutableMapping[int, str]]:
                Tuple of dictionaries mapping characters to indices and
                vice versa.
        """

        i = 1
        char_dict, ord_dict = {start_token: 0}, {0: start_token}
        for smile in mol_strings:
            for c in smile:
                if c not in char_dict:
                    char_dict[c] = i
                    ord_dict[i] = c
                    i += 1
        char_dict[pad_token], ord_dict[i] = i, pad_token
        return char_dict, ord_dict

    def _mol_string_len(self, mol_string: str) -> int:
        return len(mol_string)

    def _encode(self, mol_strings: List[str]) -> npt.NDArray[np.int32]:
        """Encodes a list of SMILES strings into a numpy array of integers."""
        encoded_mol_strings = np.empty((len(mol_strings), self._max_length))
        for mol_string_idx, mol_string in enumerate(mol_strings):
            for token_idx, token in enumerate(self.pad(mol_string, self._max_length)):
                encoded_mol_strings[mol_string_idx][token_idx] = self.token_to_index[
                    token
                ]

        return encoded_mol_strings

    def _decode(
        self, encoded_molecules: npt.NDArray[np.int32], stop_token_id: int
    ) -> List[str]:
        if len(encoded_molecules.shape) != 2:
            raise ValueError(
                f"Expected a 2D tensor of encoded molecules with shape (n_molecules, ?) but got tensor of shape {encoded_molecules.shape}"
            )

        decoded_molecules: List[str] = []
        for encoded_molecule in encoded_molecules:
            decoded_molecule = "".join(
                [self.index_to_token[ord] for ord in encoded_molecule]
            )
            decoded_molecule = self.unpad(decoded_molecule)
            decoded_molecules.append(decoded_molecule)

        return decoded_molecules
