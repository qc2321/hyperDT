import torch
from .manifolds import create_manifold

from typing import List


class ProductSpace:
    def __init__(
        self,
        dims: List[int],
        curvatures: List[float],
        agg_function: callable = lambda dists: torch.stack(dists).norm(dim=0),
    ):
        """
        agg_function should map iterables to scalars.
        For l1, you can do something like 'lambda dists: np.linalg.norm(dists, 1)
        """

        assert len(dims) == len(curvatures)
        self.dims = dims
        self.curvatures = curvatures
        self.components = [create_manifold(dim, curvature) for dim, curvature in zip(dims, curvatures)]
        self.agg_function = agg_function

    def split(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split an input tensor, columnwise, into a list of product manifold coordinates."""

        assert x.shape[1] == sum(self.dims) + len(self.dims)  # Since each manifold has an ambient dimension
        out = []
        for dim in self.dims:
            out.append(x[:, : dim + 1])
            x = x[:, dim + 1 :]
        return out

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the distance between two points on the product manifold."""

        return self.agg_function(
            [c.dist(x_i, y_i) for c, x_i, y_i in zip(self.components, self.split(x), self.split(y))]
        )
