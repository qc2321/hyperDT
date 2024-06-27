import numpy as np

def _preprocess_vecs(*vecs):
    out = []
    for v in vecs:
        v = np.array(v) # TODO: tensor
        if v.ndim == 1:
            v = v.reshape(1, -1)
        out.append(v)
    if len(out) == 1:
        return out[0]
    else:
        return out

def _minkowski_dot(u, v):
    u, v = _preprocess_vecs(u, v)
    v[:, 0] = -v[:, 0]
    return (u * v).sum(axis=-1)

def _euc_dot(u, v):
    u, v = _preprocess_vecs(u, v)
    return (u * v).sum(axis=-1)

def _euc_embed(v, curvature=0):
    v = _preprocess_vecs(v)
    timelike = np.ones(shape=(v.shape[0], 1))
    return np.concatenate([timelike, v], axis=-1)

def _hyp_embed(v, curvature=-1):
    """Compute x0 such that -x0x0 + x1x1 + x2x2 + ... = -1/K"""
    v = _preprocess_vecs(v)
    timelike = np.sqrt((v * v).sum(axis=1) + np.abs(1 / curvature)).reshape(-1, 1)
    return np.concatenate([timelike, v], axis=-1)

def _sph_embed(v, curvature=1):
    """Compute x0 such that x0x0 + x1x1 + x2x2 + ... = 1/K"""
    v = _preprocess_vecs(v)

    # Fix points that are too big
    v[np.linalg.norm(v, axis=1) > 1 / curvature] /= np.linalg.norm(
        v[np.linalg.norm(v, axis=1) > 1], axis=1, keepdims=True
    ) / curvature

    timelike = np.sqrt(1 / curvature - (v * v).sum(axis=1)).reshape(-1, 1)
    return np.concatenate([timelike, v], axis=-1)