'''
Wrapped Normal for all curvatures (modified from toy_data.py)
'''

import numpy as np
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere

# Need for bad_points:
import geomstats.backend as gs
import geomstats.algebra_utils as utils


def bad_points(points, base_points, manifold):
    """Avoid the 'Minkowski norm of 0' error by using this"""
    sq_norm_tangent_vec = manifold.embedding_space.metric.squared_norm(points)
    sq_norm_tangent_vec = gs.clip(sq_norm_tangent_vec, 0, np.inf)

    coef_1 = utils.taylor_exp_even_func(sq_norm_tangent_vec, utils.cosh_close_0, order=5)
    coef_2 = utils.taylor_exp_even_func(sq_norm_tangent_vec, utils.sinch_close_0, order=5)

    exp = gs.einsum("...,...j->...j", coef_1, base_points) + gs.einsum("...,...j->...j", coef_2, points)
    return manifold.metric.squared_norm(exp) == 0


def wrapped_normal_all_curvature(
    num_points: int,
    num_classes: int,
    noise_std: float = 1.0,
    n_dim: int = 2,
    seed: int = None,
    adjust_for_dim: bool = True,
    curvature: float = 0.0,
) -> np.ndarray:
    """Generate points from a mixture of Gaussians on the hyperboloid, Euclidean or hypersphere"""

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Select manifold type
    is_euclidean = False
    if curvature < 0.0:
        hyp = Hyperbolic(dim=n_dim)
    elif curvature > 0.0:
        hyp = Hypersphere(dim=n_dim)
    else:
        is_euclidean = True

    # Set origin for hyperboloid and hypersphere
    origin = np.array([1.0] + [0.0] * n_dim)

    # Generate random means; parallel transport from origin
    means = np.random.normal(size=(num_classes, n_dim))
    if not is_euclidean:
        means = np.concatenate(
            [
                np.zeros(shape=(num_classes, 1)),
                means,
            ],
            axis=1,
        )
        means = hyp.metric.exp(tangent_vec=means, base_point=origin)

    # Generate random covariance matrices
    covs = np.zeros((num_classes, n_dim, n_dim))
    for i in range(num_classes):
        covs[i] = np.random.normal(size=(n_dim, n_dim))
        covs[i] = covs[i] @ covs[i].T
    covs = noise_std * covs
    if adjust_for_dim:
        covs = covs / n_dim

    # Generate random class probabilities
    probs = np.random.uniform(size=num_classes)
    probs = probs / np.sum(probs)

    # First, determine class assignments
    classes = np.random.choice(num_classes, size=num_points, p=probs)

    # Sample the appropriate covariance matrix and make tangent vectors
    vecs = [np.random.multivariate_normal(np.zeros(n_dim), covs[c]) for c in classes]
    if not is_euclidean:
        tangent_vecs = np.concatenate([np.zeros(shape=(num_points, 1)), vecs], axis=1)

    # Transport each tangent vector to its corresponding mean on the hyperboloid
    if not is_euclidean:
        tangent_vecs_transported = hyp.metric.parallel_transport(
            tangent_vec=tangent_vecs, base_point=origin, end_point=means[classes]
        )

    # Exponential map to hyperboloid at the class mean
    if not is_euclidean:
        keep = ~bad_points(tangent_vecs_transported, means[classes], hyp)
        tangent_vecs_transported = tangent_vecs_transported[keep]
        classes = classes[keep]
        points = hyp.metric.exp(tangent_vec=tangent_vecs_transported, base_point=means[classes])
        points /= np.sqrt(abs(curvature))
    else:
        list_of_points = [np.random.multivariate_normal(means[c], covs[c]) for c in classes]
        points = np.asarray(list_of_points)

    return points, classes
