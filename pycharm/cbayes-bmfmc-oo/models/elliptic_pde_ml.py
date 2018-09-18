import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# The elliptic PDE model from Example 6.3 in T. Butler, J. Jakeman, and T. Wildey, ``Combining Push-Forward Measures
# and Bayes’ Rule to Construct Consistent Solutions to Stochastic Inverse Problems,'' SIAM Journal on Scientific
# Computing, vol. 40, no. 2, pp. A984–A1011, Jan. 2018.


# Dataset:
# - QoI samples are "tagged" with the key qq and are a 10000×3 array,
#   but we only consider the first QoI samples defined by the first column of this array.
# - Parameter samples are "tagged" with the key pp and are a 10000×100 array.

# QoIs:
# - Pressure values at (0.0540, 0.5487), (0.8726,0.8518) and (0.3748,0.0505)


def load_ml_data(h=10, n_models=2):
    filepath = os.path.abspath(os.path.dirname(__file__))
    qvals = []

    for i in range(n_models):
        data_set = sio.loadmat(filepath + '/elliptic_kde100_10K_h%d_q.mat' % h)
        qvals.append(data_set['qq'])
        h *= 2

    return qvals


def find_xy_pair(x, X, Y):
    idx = np.where((X == x).all(axis=1))[0][0]

    return Y[idx, :]


# Exemplary usage
if __name__ == '__main__':

    # Load data, h_min: 10, h_max: 160
    qvals = load_ml_data(h=20, n_models=4)

    # Plot data
    # plt.figure()
    # samples_lowfi = qvals[0][:, 0]
    # samples_highfi = qvals[-1][:, 0]
    # plt.plot(samples_lowfi, samples_highfi, 'r*')

    plt.figure()

    k = 0

    for i in range(len(qvals) - 1):

        # Plot data
        k += 1
        plt.subplot(int(str('33%d' % k)))
        samples_lowfi = qvals[i][:, 0]
        samples_highfi = qvals[i + 1][:, 0]
        plt.plot(samples_lowfi, samples_highfi, 'k*')

        k += 1
        plt.subplot(int(str('33%d' % k)))
        samples_lowfi = qvals[i][:, 1]
        samples_highfi = qvals[i + 1][:, 1]
        plt.plot(samples_lowfi, samples_highfi, 'b*')

        k += 1
        plt.subplot(int(str('33%d' % k)))
        samples_lowfi = qvals[i][:, 2]
        samples_highfi = qvals[i + 1][:, 2]
        plt.plot(samples_lowfi, samples_highfi, 'r*')

        # plt.figure()
        # samples_lowfi = qvals[i][:, 0]
        # samples_highfi = qvals[i][:, 2]
        # plt.plot(samples_lowfi, samples_highfi, 'g*')

    plt.show()
    exit()
