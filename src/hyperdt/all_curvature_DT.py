'''
Decision tree classifier for all curvatures
'''
import numpy as np
from hyperdt.tree import HyperbolicDecisionTreeClassifier


# Modified decision tree classifier for all curvatures
class HyperspaceDecisionTree(HyperbolicDecisionTreeClassifier):
    def __init__(self, signed_curvature=-1.0, **kwargs):
        super().__init__(**kwargs)
        self.signed_curvature = signed_curvature
        self.curvature = -signed_curvature
        self.skip_hyperboloid_check = True if signed_curvature >= 0.0 else False

    def _get_candidates(self, X, dim):
        # Hyperbolic case
        if self.signed_curvature < 0.0:
            return super()._get_candidates(X, dim)
        
        # Hypersphere case
        elif self.signed_curvature > 0.0:        
            thetas = np.arctan2(X[:, self.timelike_dim], X[:, dim])
            thetas = np.unique(thetas)      # sorted
            return (thetas[:-1] + thetas[1:]) / 2
        
        # Euclidean case
        else:
            unique_vals = np.unique(X[:, dim])  # sorted
            return (unique_vals[:-1] + unique_vals[1:]) / 2
