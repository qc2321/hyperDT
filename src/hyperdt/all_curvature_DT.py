import numpy as np
from .tree import DecisionNode, HyperbolicDecisionTreeClassifier
from .wrapped_normal_all_curvature import WrappedNormalMixture


'''
Decision tree classifier for all curvatures
'''
class HyperspaceDecisionTree(HyperbolicDecisionTreeClassifier):
    def __init__(self, signed_curvature=-1.0, **kwargs):
        super().__init__(**kwargs)
        self.signed_curvature = signed_curvature
        self.curvature = abs(signed_curvature)
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
            return np.arctan2(2, unique_vals[:-1] + unique_vals[1:])
        

'''
Class for product space object
'''
class ProductSpace:
    def __init__(self, signature=[]):
        self.signature = signature
        self.check_signature()
        self.X = []
        self.y = []
        self.means = []
    

    def check_signature(self):
        """Check if signature is valid"""
        if len(self.signature) == 0:
            raise ValueError("Signature is empty")
        for space in self.signature:
            if not isinstance(space, tuple):
                raise ValueError("Signature elements must be tuples")
            if len(space) != 2:
                raise ValueError("Signature tuples must have 2 values")
            if not isinstance(space[0], int) or space[0] <= 0:
                raise ValueError("Dimension must be a positive integer")
            if not isinstance(space[1], (int, float)):
                raise ValueError("Curvature must be an integer or float")


    def print_signature(self):
        """Print the signature of the product space"""
        for space in self.signature:
            if space[1] < 0:
                print(f"H: dim={space[0]}, K={space[1]}")
            elif space[1] > 0:
                print(f"S: dim={space[0]}, K={space[1]}")
            else:
                print(f"E: dim={space[0]}")


    def sample_clusters(self, num_points, num_classes, seed=None, cov_scale=1.0):
        """Generate data from a wrapped normal mixture on the product space"""
        classes = WrappedNormalMixture(num_points=num_points, 
                                       num_classes=num_classes).generate_class_assignments()
        for space in self.signature:
            wnm = WrappedNormalMixture(num_points=num_points, num_classes=num_classes, n_dim=space[0],
                                       curvature=space[1], seed=seed, cov_scale=cov_scale)
            means = wnm.generate_cluster_means()
            covs = [wnm.generate_covariance_matrix(wnm.n_dim, wnm.n_dim + 1, wnm.cov_scale)
                    for _ in range(wnm.num_classes)]
            points = wnm.sample_points(means, covs, classes)
            means /= np.sqrt(wnm.k) if wnm.k != 0.0 else 1.0
            self.X.append(points)
            self.y.append(classes)
            self.means.append(means)
            if wnm.curvature != 0.0:
                assert np.allclose(wnm.manifold.metric.squared_norm(points), 1 / wnm.curvature)

    def split_data(self, test_size=0.2, random_state=None):
        """Split the data into training and testing sets"""
        n = len(self.X[0])
        test_idx = np.random.choice(n, int(test_size * n), replace=False)
        self.X_train = [np.delete(X, test_idx, axis=0) for X in self.X]
        self.X_test = [X[test_idx] for X in self.X]
        self.y_train = [np.delete(y, test_idx) for y in self.y]
        self.y_test = [y[test_idx] for y in self.y]


'''
Decision tree classifier for product space
'''
class ProductSpaceDT(HyperspaceDecisionTree):
    def __init__(self, product_space: ProductSpace = None, **kwargs):
        super().__init__(**kwargs)
        self.ps = product_space
        

    def _get_space(self, dim):
        """Find the space that a dimension belongs to"""
        for i in range(len(self.ps.signature) - 1):
            if dim < sum([space[0] + 1 for space in self.ps.signature[:i+1]]):
                return i
        return len(self.ps.signature) - 1


    def _fit_node(self, X, y, depth):
        """Recursively fit a node of the tree. Modified from DecisionTreeClassifier
        to iterate over all dimensions across different manifolds."""
        # Base case
        if depth == self.max_depth or len(y) <= self.min_samples_split or len(np.unique(y)) == 1:
            value, probs = self._leaf_values(y)
            return DecisionNode(value=value, probs=probs)

        # Recursively find the best split:
        best_dim, best_theta, best_score = None, None, -1
        for dim in self.dims_ex_time:
            space = self._get_space(dim)
            self.signed_curvature = self.ps.signature[space][1]
            dim_in_space = dim - self.timelike_dims[space]
            for count, theta in enumerate(self._get_candidates(X=X[space], dim=dim_in_space)):
                left, right = self._get_split(X=X[space], dim=dim_in_space, theta=theta)
                min_len = np.min([len(y[left]), len(y[right])])
                if min_len >= self.min_samples_leaf:
                    score = self._information_gain(left, right, y)
                    if score >= best_score + self.min_impurity_decrease:
                        best_dim, best_theta, best_score = dim, theta, score

        # Fallback case:
        if best_score == -1:
            value, probs = self._leaf_values(y)
            return DecisionNode(value=value, probs=probs)

        # Populate:
        node = DecisionNode(feature=best_dim, theta=best_theta)
        node.score = best_score
        best_space = self._get_space(best_dim)
        best_dim_in_space = best_dim - self.timelike_dims[best_space]
        left, right = self._get_split(X=X[best_space], dim=best_dim_in_space, theta=best_theta)
        X_left = [X[space][left] for space in range(len(self.ps.signature))]
        X_right = [X[space][right] for space in range(len(self.ps.signature))]
        node.left = self._fit_node(X=X_left, y=y[left], depth=depth + 1)
        node.right = self._fit_node(X=X_right, y=y[right], depth=depth + 1)
        return node
    

    def fit(self):
        """Fit a decision tree to the data. Modified from HyperbolicDecisionTreeClassifier
        to remove multiple timelike dimensions in product space."""
        # Find all dimensions in product space (including timelike dimensions)
        self.all_dims = list(range(sum([space[0] + 1 for space in self.ps.signature])))
        
        # Find indices of timelike dimensions in product space
        self.timelike_dims = [0]
        for i in range(len(self.ps.signature) - 1):
            self.timelike_dims.append(sum([space[0] + 1 for space in self.ps.signature[:i+1]]))
        
        # Remove timelike dimensions from list of dimensions
        self.dims_ex_time = list(np.delete(np.array(self.all_dims), self.timelike_dims))
        self.classes_ = np.unique(self.ps.y_train[0])

        # Call recursive fitting function
        self.tree = self._fit_node(X=self.ps.X_train, y=self.ps.y_train[0], depth=0)


    def predict(self, X):
        """Predict labels for samples in X"""
        return np.array([self.classes_[self._traverse(x).value] for x in X])
