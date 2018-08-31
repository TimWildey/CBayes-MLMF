import numpy as np

# A simple polynomial type forward map


def lambda_p(lam, p):
    return lam ** p + 0.01*np.random.randn(np.shape(lam)[0])


def get_prior_samples(n_samples):
    return np.random.uniform(low=-1, high=1, size=(n_samples, 1))