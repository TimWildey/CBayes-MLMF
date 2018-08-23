import numpy as np


def rejection_sampling(r):
    # Perform accept/reject sampling on a set of proposal samples using
    # the weights r associated with the set of samples and return
    # the indices idx of the proposal sample set that are accepted.
    n = r.size  # size of proposal sample set
    check = np.random.uniform(low=0, high=1, size=n)  # create random uniform weights to check r against
    m = np.max(r)
    new_r = r/m  # normalize weights
    idx = np.where(new_r >= check)[0]  # rejection criterion
    return idx