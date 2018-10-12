import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# The elliptic PDE model from Example 6.3 in T. Butler, J. Jakeman, and T. Wildey, ``Combining Push-Forward Measures
# and Bayes’ Rule to Construct Consistent Solutions to Stochastic Inverse Problems,'' SIAM Journal on Scientific
# Computing, vol. 40, no. 2, pp. A984–A1011, Jan. 2018.


# Dataset:
# - QoI samples are "tagged" with the key qq and are a 50000×4 array,
#   but we only consider the first three QoI samples.

# QoIs:
# - Pressure values at (0.0540, 0.5487), (0.8726,0.8518) and (0.3748,0.0505)


def load_data():
    filepath = os.path.abspath(os.path.dirname(__file__))
    qvals = []
    h = 40
    n_models = 3

    for i in range(n_models):
        data_set = sio.loadmat(filepath + '/elliptic_kde100_50K_h%d_q.mat' % h)
        qvals.append(data_set['qq'][:, :3])
        h *= 2

    return qvals


def find_xy_pair(x, X, Y):
    idx = np.where((X == x).all(axis=1))[0][0]

    return Y[idx, :]


# Exploring the data
if __name__ == '__main__':

    # Load data
    qvals = load_data()

    n_samples = int(1e4)

    lf = qvals[0][:n_samples] ** 1.2
    mf = qvals[1][:n_samples] ** 1.1
    hf = qvals[2][:n_samples]

    print('Q1 HF median: %f' % np.median(hf[:, 0]))
    print('Q2 HF median: %f' % np.median(hf[:, 1]))
    print('Q3 HF median: %f' % np.median(hf[:, 2]))

    plt.figure()
    plt.subplot(131)
    plt.plot(lf[:, 0], lf[:, 1], 'k*')
    plt.subplot(132)
    plt.plot(lf[:, 0], lf[:, 2], 'k*')
    plt.subplot(133)
    plt.plot(lf[:, 1], lf[:, 2], 'k*')

    print('Q1-Q2 correlations: %f' % np.corrcoef(lf[:, 0], lf[:, 1])[0, 1])
    print('Q1-Q3 correlations: %f' % np.corrcoef(lf[:, 0], lf[:, 2])[0, 1])
    print('Q2-Q3 correlations: %f' % np.corrcoef(lf[:, 1], lf[:, 2])[0, 1])

    print('Q1: q_0-Q correlations: %f' % np.corrcoef(lf[:, 0], hf[:, 0])[0, 1])
    print('Q2: q_1-Q correlations: %f' % np.corrcoef(mf[:, 0], hf[:, 0])[0, 1])
    print('Q3: q_0-q_1 correlations: %f' % np.corrcoef(lf[:, 0], mf[:, 0])[0, 1])

    plt.figure()

    k = 0

    for i in range(len(qvals) - 1):
        if i is 0:
            print('lf-mf correlations')
            samples_lowfi = lf
            samples_highfi = mf
        elif i is 1:
            print('mf-hf correlations')
            samples_lowfi = mf
            samples_highfi = hf

        # Plot data
        k += 1
        plt.subplot(int(str('23%d' % k)))
        plt.plot(samples_lowfi[:, 0], samples_highfi[:, 0], 'b*')
        print(np.corrcoef(samples_lowfi[:, 0], samples_highfi[:, 0])[0, 1])

        k += 1
        plt.subplot(int(str('23%d' % k)))
        plt.plot(samples_lowfi[:, 1], samples_highfi[:, 1], 'r*')
        print(np.corrcoef(samples_lowfi[:, 1], samples_highfi[:, 1])[0, 1])

        k += 1
        plt.subplot(int(str('23%d' % k)))
        plt.plot(samples_lowfi[:, 2], samples_highfi[:, 2], 'g*')
        print(np.corrcoef(samples_lowfi[:, 2], samples_highfi[:, 2])[0, 1])

    plt.show()
    exit()
