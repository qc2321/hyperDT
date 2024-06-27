import numpy as np

from .utils import _minkowski_dot, _euc_dot, _euc_embed, _hyp_embed, _sph_embed, _preprocess_vecs

class GenericManifold():
    def __init__(self, dim, curvature):
        self.dim = int(dim)
        self.curvature = float(curvature)
    
    def _preprocess_for_dist(u, v) -> bool:
        u, v = _preprocess_vecs(u, v)
        return (
            u.shape[1] == v.shape[1] == self.dim + 1 and self.on_manifold(u) and self.on_manifold(v)
        )
    
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
        return np.allclose(x[:,0], 1.) and x.shape[1] == self.dim + 1
    
    def dist(self, u: np.ndarray, v: np.ndarray) -> float:
        u, v = _preprocess_for_dist(u, v)
        return np.linalg.norm(u[None, :] - v[:, None], axis=2)
    
class Hyperbolic(GenericManifold):
    def embed_point(self, x):
        return _hyp_embed(x, self.curvature)

    def on_manifold(self, x):
        x = _preprocess_vecs(x)
        return (
            np.allclose(_minkowski_dot(x, x), 1 / self.curvature) and (x[:,0] > 0).all() and x.shape[1] == self.dim + 1
        )
    
    def dist(self, u: np.ndarray, v: np.ndarray) -> float:
        u, v = _preprocess_for_dist(u, v)
        return np.arccosh(self.curvature * _minkowski_dot(u[None, :], v[:, None])) / np.sqrt(-self.curvature)

class Spherical(GenericManifold):
    def embed_point(self, x):
        return _sph_embed(x, self.curvature)

    def on_manifold(self, x):
        x = _preprocess_vecs(x)
        return np.allclose(_euc_dot(x, x), 1 / self.curvature) and x.shape[1] == self.dim + 1

    def dist(self, u: np.ndarray, v: np.ndarray) -> float:
        u, v = _preprocess_for_dist(u, v)
        # TODO: is this even correct?
        return np.arccos(self.curvature * _euc_dot(u[None, :], v[:, None])) / np.sqrt(self.curvature)
    

class ComponentManifold():
    def __init__(self, dim:int, curvature:float):
        if curvature < 0:
            self = Hyperbolic(dim, curvature)
        elif curvature == 0:
            self = Euclidean(dim, curvature)
        elif curvature > 0:
            self = Spherical(dim, curvature)
