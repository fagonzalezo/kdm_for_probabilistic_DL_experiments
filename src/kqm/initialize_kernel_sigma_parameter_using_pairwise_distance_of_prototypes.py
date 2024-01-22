import numpy as np
from sklearn.metrics import pairwise_distances

def initialize_kernel_sigma_parameter_using_pairwise_distance_of_prototypes(encoded_samples, kernel):
    distances = pairwise_distances(encoded_samples.numpy())
    sigma = np.mean(distances) 
    kernel.sigma.assign(sigma) 
    print(f"Initial sigma: {kernel.sigma.numpy()}")


