import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.euclidean import Euclidean
from scipy.stats import wishart


'''
Wrapped normal mixture class that generates points from a mixture of Gaussians on
the hyperboloid, Euclidean or hypersphere with defined curvature.
'''
class WrappedNormalMixture:
    def __init__(
        self,
        num_points: int,
        num_classes: int,
        n_dim: int = 2,
        curvature: float = 0.0,
        seed: int = None,
        cov_scale: float = 0.3,
    ):
        self.num_points = num_points
        self.num_classes = num_classes
        self.n_dim = n_dim
        self.curvature = curvature
        self.k = abs(curvature)
        self.curv_sign = 1
        self.seed = seed
        self.cov_scale = cov_scale

        # Set random number generator
        self.rng = np.random.default_rng(self.seed)
        
        # Set manifold based on curvature
        if curvature == 0.0:
            self.manifold = Euclidean(dim=n_dim)
        elif curvature > 0.0:
            self.manifold = Hypersphere(dim=n_dim)
        else:
            self.manifold = Hyperboloid(dim=n_dim)
            self.curv_sign = -1

        # Set origin for hyperboloid and hypersphere
        self.origin = np.array([1.0] + [0.0] * self.n_dim)
        
    
    def generate_cluster_means(self):
        '''
        Generate random cluster means on the manifold, adjusted for curvature.
        '''
        means = np.concatenate(
            (
                np.zeros(shape=(self.num_classes, 1)),
                self.rng.normal(size=(self.num_classes, self.n_dim)),
            ),
            axis=1,
        )
        # Adjust for curvature
        means *= np.sqrt(self.k) if self.k != 0.0 else 1.0

        return self.manifold.metric.exp(tangent_vec=means, base_point=self.origin)


    def generate_covariance_matrix(self, dims, deg_freedom, scale):
        '''
        Generate random covariance matrix based on Wishart distribution.
        '''
        scale_matrix = scale * np.eye(dims)
        cov_matrix = wishart.rvs(df=deg_freedom, scale=scale_matrix, random_state=self.rng)

        return cov_matrix


    def generate_class_assignments(self):
        '''
        Generate random class assignments based on uniform class probabilities.
        '''
        probs = self.rng.uniform(size=self.num_classes)
        probs = probs / np.sum(probs)

        return self.rng.choice(self.num_classes, size=self.num_points, p=probs)


    def sample_points(self, means, covs, classes):
        '''
        Generate random samples for each cluster based on the cluster means and covariance matrices.
        '''
        # Generate random vectors on tangent plane for each class
        vecs = np.array([self.rng.multivariate_normal(np.zeros(self.n_dim), covs[c]) for c in classes])
        
        # Adjust for curvature and prepend zeros for ambient space
        vecs *= np.sqrt(self.k) if self.k != 0.0 else 1.0
        vecs = np.column_stack((np.zeros(vecs.shape[0]), vecs))

        # Parallel transport vectors from origin to sampled means on the manifold
        tangent_vecs = self.manifold.metric.parallel_transport(vecs, self.origin, end_point=means[classes])

        # Exponential map to manifold at the class mean
        points = self.manifold.metric.exp(tangent_vec=tangent_vecs, base_point=means[classes])
        
        # Adjust for curvature
        points /= np.sqrt(abs(self.k)) if self.k != 0.0 else 1.0

        return points


    def generate_data(self):
        '''
        Generate Gaussian mixture data.
        '''
        # Generate random class means on the manifold
        means = self.generate_cluster_means()
        
        # Generate covariance matrices for each class
        covs = [self.generate_covariance_matrix(self.n_dim, self.n_dim + 1, self.cov_scale) for _ in range(self.num_classes)]

        # Generate class assignments
        classes = self.generate_class_assignments()
        
        # Sample points from the Gaussian mixture
        points = self.sample_points(means, covs, classes)

        # Readjust means for curvature
        means /= np.sqrt(self.k) if self.k != 0.0 else 1.0

        return points, classes, means
