import typing as t
from typing import Optional, Sequence

import torch
from torch import nn

from ..distance import CodebookDistance, EuclideanCodebookDistance


class VectorQuantizer(nn.Module):
    """The VectorQuantizer layer quantizes a batch of input vectors by replacing each
    vector with the closest vector from an internal codebook. The closest vector is
    determined using the a specified distance function.

    Methods:
        compute_distance: compute a matrix of pair-wise distances between row vectors in
            tensor t1 and tensor t2.
        quantize: quantizes a given set of input vectors by replacing each input vector
            with the closest vector from the codebook.

    Inputs:
        torch.Tensor: a batch of input vectors. The VectorQuantizer layer expects to
            operate one tensors of shape (B, d0, d1, ..., code_dim). You can specify a
            permutation order to ensure that the VectorQuantizer can handle inputs where
            the embedding dimension may not be the final one.

    Outputs:
        torch.Tensor: a batch of quantized vectors. The VectorQuantizer layer returns a
            tensor of shape (B, d0, d1, ..., code_dim) if no permutation order is
            specified. If a permutation order is specified, the VectorQuantizer layer
            returns a tensor of the same shape as the input tensor, following an inverse
            permutation operation from (B, d0, d1, ..., code_dim) to the original input
            dimension ordering.
    """

    def __init__(
        self,
        n_codes: int,
        code_dim: int,
        permute_order: Optional[Sequence[int]],
        distance_fn: Optional[CodebookDistance] = None,
        padding_token_index: Optional[int] = None,
        embedding_weights: Optional[torch.Tensor] = None,
    ):
        """Initializes the VectorQuantizer layer.

        Args:
            code_dim (int): the dimension of each code / embedding vector.
            permute_order (Sequence[int], optional): permutation order to apply to the
                input tensor. An inverse permutation order is applied to the output
                tensor to ensure that the output tensor has the same shape as the input
                tensor. If no permutation is needed, set this to ``None``. Because the
                quantizer expects to operate on tensors of shape (B, d0, d1, ...,
                embedding_dim) this parameter is needed to ensure the quantizer can
                handle inputs where the embedding dimension may not be the final one.
                For example, if the output of the encoder is a tensor of shape (B,
                embedding_dim, d0, d1) then the permutation order should be (0, 2, 3,
                1).
            distance_fn (_CodebookDistance, optional): function to compute distances
                between encoded vectors and codebook vectors. If None, defaults to
                EuclidianCodebookDistance().
            padding_token_index (int, optional): index of the padding token in the
                codebook. Defaults to None.
            embedding_weights (torch.Tensor, optional): pre-trained codebook weights.
                Defaults to None.
        """
        super(VectorQuantizer, self).__init__()
        distance_fn = (
            EuclideanCodebookDistance() if distance_fn is None else distance_fn
        )

        self.n_codes = n_codes
        self.code_dim = code_dim

        self.codebook = nn.Embedding(
            self.n_codes, self.code_dim, padding_idx=padding_token_index
        )
        self.distance_fn = distance_fn

        if embedding_weights is None:
            self.codebook.weight.data.uniform_(-1 / self.n_codes, 1 / self.n_codes)
        else:
            self.codebook.weight.data.copy_(embedding_weights)

        self.permute_order = permute_order
        self.inverse_permute_order = (
            self._inverse_permutation_order(permute_order)
            if permute_order is not None
            else None
        )

    @staticmethod
    def _inverse_permutation_order(permutation_order: Sequence[int]) -> Sequence[int]:
        """Given a permutation order, return the order needed to return the element
        along each dimension to its original location.

        Example:
            >>> permute_order = [1, 0]
            >>> print(_inverse_permute_order(permute_order))
            >>> [0, 1]
            >>> permute_order = [2, 0, 1]
            >>> print(_inverse_permute_order(permute_order))
            >>> [1, 2, 0]
            >>> permute_order = [0, 1, 2]
            >>> print(_inverse_permute_order(permute_order))
            >>> [0, 1, 2]
            >>> import torch
            >>> t = torch.rand(1, 2, 3)
            >>> print(t.permute([2, 0, 1]).shape)
            >>> torch.Size([3, 1, 2])
            >>> print(t.permute(_inverse_permute_order([2, 0, 1])).shape)
            >>> torch.Size([1, 2, 3])

        Args:
            permutation_order (Sequence[int]): a sequence of integers representing the
                order of dimensions to permute the input tensor.

        Returns:
            Sequence[int]: a sequence of integers representing the order of dimensions
                allowing to "undo" the permutation defined by the input permutation
                order.
        """
        # map taking dimension to its new index after permutation
        # i.e. for permutation_order = [2, 0, 1], dim_to_index_map = {2: 0, 0: 1, 1: 2}
        dim_to_index_map = {dim: index for index, dim in enumerate(permutation_order)}

        # if element at position 2 is moved to position 0, then to undo that operation
        # we need to move the element at position 0 to position 2.
        # i.e. for permutation_order = [2, 0, 1], inverse_permute_order = [1, 2, 0]
        inverse_permute_order = [
            dim_to_index_map[dim] for dim in range(len(permutation_order))
        ]

        return inverse_permute_order

    def compute_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Compute a matrix of pair-wise distances between vectors in tensor t1 and
        tensor t2.

        Returns:
            torch.Tensor: a 2D matrix of pairwise distances of shape (n_vec_t1,
                n_vec_t2). The value at index (i, j) in the output matrix is the
                distance between the vector at index i in t1 and the vector at index j
                in t2.
        """
        return self.distance_fn(t1, t2)

    def forward(self, inputs: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """Quantizes a given set of input vectors by replacing each input vector
        with the closest vector from the codebook. The closest vector is determined
        using the user-specified distance function.

        Args:
            inputs (torch.Tensor): input tensor of shape (B, dim1, dim2, ..., code_dim).
                Note that the last dimension must be equal to the embedding dimension.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: quantized vectors with shape (B, dim1,
                dim2, ..., code_dim) and a matrix of one-hot encoded codebook indices of
                shape (B, dim1 * dim2 * ..., )

        Raises:
            RuntimeError: if the final entry in the shape of the input tensor is not
            equal to the embedding dimension.
        """
        # input shape (B, dim1, dim2, ... )
        if self.permute_order is not None:
            inputs = inputs.permute(*self.permute_order).contiguous()

        input_shape = inputs.shape
        if input_shape[-1] != self.code_dim:
            raise ValueError(
                "VectorQuantizer expects input tensors to have shape (B, dim1, dim2, "
                "..., code_dim), however the final entry of the input tensor shape is "
                "not equal to the embedding dimension. {} != {}".format(
                    input_shape[-1], self.code_dim
                )
            )

        # flatten inputs to have shape (B * dim1 * dim2 * ..., code_dim)
        flat_input = inputs.view(-1, self.code_dim)

        # pairwise distance matrix, where m[i][j] is the distance between
        # input vector at index i and codebook vector at index j
        distance_matrix = self.compute_distance(flat_input, self.codebook.weight)

        # find the closest codebook vector to each input vector (n_input_vectors, 1)
        # pylint: disable=no-member
        code_indices = torch.argmin(distance_matrix, dim=1).unsqueeze(1)

        # placeholder for the quantized encodings
        one_hot_code_indices = torch.zeros(  # pylint: disable=no-member
            code_indices.shape[0], self.n_codes, device=inputs.device
        )
        one_hot_code_indices.scatter_(1, code_indices, 1)

        # Quantize and unflatten
        # pylint: disable=no-member
        quantized = torch.matmul(one_hot_code_indices, self.codebook.weight)
        quantized = quantized.view(input_shape)

        if self.inverse_permute_order is not None:
            quantized = quantized.permute(*self.inverse_permute_order).contiguous()

        return quantized, one_hot_code_indices

    def quantize(self, inputs: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """Quantizes a given set of input vectors by replacing each input vector
        with the closest vector from the codebook. The closest vector is determined
        using the user-specified distance function.

        Args:
            inputs (torch.Tensor): input tensor of shape (B, dim1, dim2, ..., code_dim).
                Note that the last dimension must be equal to the embedding dimension.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: quantized vectors with shape (B, dim1,
                dim2, ..., code_dim) and a matrix of one-hot encoded codebook indices of
                shape (B, dim1 * dim2 * ..., )

        Raises:
            RuntimeError: if the final entry in the shape of the input tensor is not
                equal to the embedding dimension.

        Note:
            This method is equivalent to calling ``forward`` or
            ``VectorQuantizer(...)(data)``.
        """
        return self(inputs)
