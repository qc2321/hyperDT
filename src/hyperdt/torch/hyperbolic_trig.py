import torch


def _dist(x1, x2):
    """
    Closed form for distance between two unique rotational intersections where all other
    coordinates are 0

    A few observations and simplifications:
    - The distance is defined as arccosh(-a(theta1) * a(theta2) * cos(theta1 + theta2))
        - a(theta) = sqrt(-sec(2theta))
    - All angles are in (pi / 4, 3pi / 4). As such:
        - sec(theta) > 0, so we never take sqrt of a negative number
        - cos(theta1 + theta2) < 0, so we always take arccosh of a positive number
    - We can simplify the distance to a(theta1) * a(theta2) * cos(theta1 + theta2)
        - This is more stable than taking arccosh
    """
    a1 = torch.sqrt(-1 / torch.cos(2 * x1))
    a2 = torch.sqrt(-1 / torch.cos(2 * x2))
    dist = a1 * a2 * torch.cos(x1 + x2)

    # Deal with really close values - numerical weirdness
    if torch.abs(dist) < 1e-6 or torch.abs(x1 - x2) < 1e-6:
        dist = 0
    return dist


def _dist_aberration(m, x1, x2):
    """This is 0 when d(theta1, m) = d(theta2, m) = d(theta1, theta2)/2"""
    return _dist(x1, m) - _dist(m, x2)


def _hyperbolic_midpoint(a, b):
    """New method: analytical closed forms for hyperbolic midpoint"""
    if torch.isclose(a, b):
        return a
    v = torch.sin(2 * a - 2 * b) / (torch.sin(a + b) * torch.sin(b - a))
    coef = -1 if a < torch.pi - b else 1
    sol = (-v + coef * torch.sqrt(v**2 - 4)) / 2
    return torch.arctan2(torch.tensor(1), sol) % torch.pi


def get_midpoint(theta1, theta2, skip_checks=True, method="hyperbolic"):
    """Find hyperbolic midpoint of two angles"""
    if method == "hyperbolic":
        root = _hyperbolic_midpoint(theta1, theta2)

    elif method == "bisect":
        root = (theta1 + theta2) / 2

    else:
        raise ValueError(f"Unknown method {method}")

    if not skip_checks:
        assert torch.abs(_dist_aberration(root, theta1, theta2)) < 1e-6
        assert root >= theta_min and root <= theta_max

    return root


def get_candidates(X, dim, timelike_dim, method="hyperbolic", cache=None):
    """Get candidate split points for hyperbolic decision tree"""
    thetas = torch.arctan2(X[:, timelike_dim], X[:, dim])
    thetas = torch.unique(thetas)  # This also sorts

    # Get all pairs of angles
    # func = cache.cache_decorator(get_midpoint) if cache is not None else get_midpoint
    # candidates = torch.tensor([func(theta1, theta2, method=method) for theta1, theta2 in zip(thetas[:-1], thetas[1:])])
    candidates = torch.tensor(
        [get_midpoint(theta1, theta2, method=method) for theta1, theta2 in zip(thetas[:-1], thetas[1:])],
        device=X.device,
    )
    return candidates
