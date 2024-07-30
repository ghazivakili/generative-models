import torch

from .base import CodebookDistance


class EuclideanCodebookDistance(CodebookDistance):
    """Computes a Euclidean pairwise distance between every vector in 2D tensors xx and yy."""

    def measure(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Computes a Euclidean pairwise distance between every vector in 2D tensors xx and yy.

        Args:
            xx (torch.Tensor): tensor with shape (n_x, vector_dim)
            yy (torch.Tensor): tensor with shape (n_y, vector_dim)

        Returns:
            torch.Tensor: tensor with pairwise distance between every vector in x and y
                of shape (n_x, n_y) where the ij-th entry is the Euclidean distance
                between vectors xx{row=i} and yy{row=j}.
        """
        xx_sq = torch.sum(xx**2, dim=1, keepdim=True)  # (n_x,)
        yy_sq = torch.sum(yy**2, dim=1)  # (n_y,)
        xx_yy = torch.matmul(xx, yy.t())  # (n_x, n_y)

        # when we sum x_sq and y_sq we get a (n_x, n_y) matrix
        return xx_sq + yy_sq - 2 * xx_yy
