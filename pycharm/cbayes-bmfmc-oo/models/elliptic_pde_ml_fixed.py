import os
import numpy as np
import scipy.io as sio


# The elliptic PDE model from Example 6.3 in T. Butler, J. Jakeman, and T. Wildey, ``Combining Push-Forward Measures
# and Bayes’ Rule to Construct Consistent Solutions to Stochastic Inverse Problems,'' SIAM Journal on Scientific
# Computing, vol. 40, no. 2, pp. A984–A1011, Jan. 2018.


# Dataset:
# - QoI samples are "tagged" with the key qq and are a 10000×3 array,
#   but we only consider the first QoI samples defined by the first column of this array.
# - Parameter samples are "tagged" with the key pp and are a 10000×100 array.

# QoIs:
# - Pressure values at (0.0540, 0.5487), (0.8726,0.8518) and (0.3748,0.0505)


def load_data(h=10, n_evals=[50]):
    filepath = os.path.abspath(os.path.dirname(__file__))
    qvals = []

    for i in range(len(n_evals)+1):
        data_set = sio.loadmat(filepath + '/elliptic_kde100_10K_h%d_q.mat' % h)
        if i == 0:
            qvals.append(data_set['qq'])
        else:
            qvals.append(data_set['qq'][0:n_evals[i-1], :])
        h *= 2

    return qvals


def load_mc_reference(h=160):

    filepath = os.path.abspath(os.path.dirname(__file__))
    data_set = sio.loadmat(filepath + '/elliptic_kde100_10K_h%d_q.mat' % h)

    return data_set['qq']


def find_xy_pair(x, X, Y):
    idx = np.where((X == x).all(axis=1))[0][0]

    return Y[idx, :]


# Exemplary usage
if __name__ == '__main__':

    # Load data, h_min: 10, h_max: 160
    qvals = load_data(h=20, n_evals=[200, 50, 10])
