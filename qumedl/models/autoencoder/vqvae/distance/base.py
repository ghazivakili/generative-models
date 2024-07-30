from abc import ABC, abstractmethod

import torch


class CodebookDistance(ABC):
    """Abstract base class for distance measures.
    Implements a distance measure between individual vectors from two tensors.
    Given two 2D tensors xx and yy of shape (n_x, vector_dim) and (n_y, vector_dim) respectively,
    the ij-th entry of the tensor returned by this class is a distance measure between
    the between vectors xx{row=i} and yy{row=j}.
    """

    def __call__(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        return self.measure(xx, yy)

    @abstractmethod
    def measure(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Computes a distance measure between every vector in 2D tensors xx and yy.

        Args:
            xx (torch.Tensor): a tensor with shape (n_x, vector_dim)
            yy (torch.Tensor): a tensor with shape (n_y, vector_dim)

        Returns:
            torch.Tensor: a tensor with shape (n_x, n_y) where the ij-th entry is the
                distance measure between vectors xx{row=i} and yy{row=j}.
        """
        raise NotImplementedError
