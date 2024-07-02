import torch
from math import sqrt

from .utils import _minkowski_dot, _euc_dot, _euc_embed, _hyp_embed, _sph_embed, _preprocess_vecs


class GenericManifold:
    def __init__(self, dim, curvature):
        self.dim = int(dim)
        self.curvature = float(curvature)

    def _preprocess_for_dist(self, u, v) -> bool:
        u, v = _preprocess_vecs(u, v)
        assert u.shape[1] == v.shape[1] == self.dim + 1
        assert self.on_manifold(u) and self.on_manifold(v)
        return u, v

    def embed_point(self, x):
        raise NotImplementedError

    def on_manifold(self, x):
        raise NotImplementedError

    def dist(self, u, v):
        raise NotImplementedError


class Euclidean(GenericManifold):
    def embed_point(self, x):
        return _euc_embed(x)

    def on_manifold(self, x):
        x = _preprocess_vecs(x)
        return torch.allclose(x[:, 0], torch.Tensor([1.0])) and x.shape[1] == self.dim + 1

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        u, v = self._preprocess_for_dist(u, v)
        return torch.linalg.vector_norm(u[None, :] - v[:, None], axis=2)


class Hyperbolic(GenericManifold):
    def embed_point(self, x):
        return _hyp_embed(x, self.curvature)

    def on_manifold(self, x):
        x = _preprocess_vecs(x)
        return (
            torch.allclose(_minkowski_dot(x, x), torch.Tensor([1 / self.curvature]))
            and (x[:, 0] > 0).all()
            and x.shape[1] == self.dim + 1
        )

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        u, v = self._preprocess_for_dist(u, v)
        return torch.arccosh(self.curvature * _minkowski_dot(u[None, :], v[:, None])) / sqrt(-self.curvature)


class Spherical(GenericManifold):
    def embed_point(self, x):
        return _sph_embed(x, self.curvature)

    def on_manifold(self, x):
        x = _preprocess_vecs(x)
        return torch.allclose(_euc_dot(x, x), torch.Tensor([1 / self.curvature])) and x.shape[1] == self.dim + 1

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        u, v = self._preprocess_for_dist(u, v)
        # TODO: is this even correct?
        return torch.arccos(self.curvature * _euc_dot(u[None, :], v[:, None])) / sqrt(self.curvature)


def create_manifold(dim, curvature):
    if curvature < 0:
        return Hyperbolic(dim, curvature)
    elif curvature == 0:
        return Euclidean(dim, curvature)
    elif curvature > 0:
        return Spherical(dim, curvature)
