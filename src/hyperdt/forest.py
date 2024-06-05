"""Hyperbolic random forest"""

import numpy as np
from scipy import stats

from tqdm import tqdm

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from .tree import (
    DecisionTreeClassifier,
    HyperbolicDecisionTreeClassifier,
    DecisionTreeRegressor,
    HyperbolicDecisionTreeRegressor,
)
from .cache import SplitCache
from .product_space_DT import ProductSpaceDT


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
        weights=None,
        n_jobs=-1,
        tree_type=DecisionTreeClassifier,
        random_state=None,
        skip_hyperboloid_check=False,
        angle_midpoint_method="hyperbolic",
    ):
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        self.tree_params = {}
        self.max_depth = self.tree_params["max_depth"] = max_depth
        self.min_samples_split = self.tree_params["min_samples_split"] = min_samples_split
        self.min_samples_leaf = self.tree_params["min_samples_leaf"] = min_samples_leaf
        self.criterion = self.tree_params["criterion"] = criterion
        self.weights = self.tree_params["weights"] = weights
        self.cache = self.tree_params["cache"] = SplitCache()
        self.skip_hyperboloid_check = self.tree_params["skip_hyperboloid_check"] = skip_hyperboloid_check
        self.angle_midpoint_method = self.tree_params["angle_midpoint_method"] = angle_midpoint_method

        self.tree_type = tree_type
        self.trees = self._get_trees()
        self.random_state = random_state

        assert isinstance(self.trees[0], self.tree_type), "Tree type mismatch"

    def _get_trees(self):
        return [self.tree_type(**self.tree_params) for _ in range(self.n_estimators)]

    def _generate_subsample(self, X, y):
        """Generate a random subsample of the data"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y, use_tqdm=False, seed=None):
        """Fit a decision tree to subsamples"""
        self.classes_ = np.unique(y)

        if seed is not None:
            self.random_state = seed
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Fit decision trees individually (parallelized):
        trees = tqdm(self.trees) if use_tqdm else self.trees
        if self.n_jobs != 1:
            fitted_trees = Parallel(n_jobs=self.n_jobs)(
                delayed(tree.fit)(*self._generate_subsample(X, y)) for tree in trees
            )
            self.trees = fitted_trees
        else:
            for tree in trees:
                X_sample, y_sample = self._generate_subsample(X, y)
                tree.fit(X_sample, y_sample)
        return self

    def predict(self, X):
        """Predict the class of each sample in X"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return stats.mode(predictions, axis=0, keepdims=False)[0]

    def predict_proba(self, X):
        """Predict the class probabilities of each sample in X"""
        predictions = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        return np.mean(self.predict(X) == y)


class RandomForestRegressor(RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(tree_type=DecisionTreeRegressor, **kwargs)
        assert isinstance(self.trees[0], DecisionTreeRegressor)


class HyperbolicRandomForestClassifier(RandomForestClassifier):
    def __init__(self, timelike_dim=0, curvature=-1, **kwargs):
        super().__init__(tree_type=HyperbolicDecisionTreeClassifier, **kwargs)
        self.curvature = np.abs(curvature)
        for tree in self.trees:
            tree.curvature = np.abs(curvature)
        self.timelike_dim = self.tree_params["timelike_dim"] = timelike_dim
        assert isinstance(self.trees[0], HyperbolicDecisionTreeClassifier)


class HyperbolicRandomForestRegressor(RandomForestClassifier):
    def __init__(self, timelike_dim=0, curvature=-1, **kwargs):
        super().__init__(tree_type=HyperbolicDecisionTreeRegressor, **kwargs)
        self.curvature = np.abs(curvature)
        for tree in self.trees:
            tree.curvature = np.abs(curvature)
        self.timelike_dim = self.tree_params["timelike_dim"] = timelike_dim
        assert isinstance(self.trees[0], HyperbolicDecisionTreeRegressor)

class ProductSpaceRF(BaseEstimator, ClassifierMixin):
    def __init__(self, product_space, n_estimators=100, max_features='sqrt', max_samples=1.0, random_state=None, max_depth=3):
        self.product_space = product_space
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.trees = [ProductSpaceDT(product_space=product_space, max_depth=max_depth) for _ in range(n_estimators)]

    def _generate_subsample(self, X, y):
        n_samples = X.shape[0]
        sample_size = int(n_samples * self.max_samples)
        indices = np.random.choice(n_samples, size=sample_size, replace=True)
        return X[indices], y[indices]

    # def _generate_subfeatures(self, X):
    #     n_features = X.shape[1]
    #     if self.max_features == 'sqrt':
    #         feature_size = int(np.sqrt(n_features))
    #     elif self.max_features == 'log2':
    #         feature_size = int(np.log2(n_features))
    #     else:
    #         feature_size = int(n_features * self.max_features)
    #     feature_indices = np.random.choice(n_features, size=feature_size, replace=False)
    #     return feature_indices

    def fit(self):
        for tree in self.trees:
            X_sample, y_sample = self._generate_subsample(tree.ps.X_train, tree.ps.y_train)
            tree.ps.X_train = X_sample
            tree.ps.y_train = y_sample
            # feature_indices = self._generate_subfeatures(X_sample)
            tree.fit()
        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # predictions = np.array([tree.predict(X[:, tree.feature_indices_]) for tree in self.trees])
        return stats.mode(predictions, axis=0, keepdims=False)[0]

    def predict_proba(self, X):
        predictions = np.array([tree.predict_proba(X) for tree in self.trees])
        # predictions = np.array([tree.predict_proba(X[:, tree.feature_indices_]) for tree in self.trees])
        return np.mean(predictions, axis=0)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)