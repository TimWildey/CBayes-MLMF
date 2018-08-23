import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern


# The elliptic PDE model from Example 6.3 in T. Butler, J. Jakeman, and T. Wildey, ``Combining Push-Forward Measures
# and Bayes’ Rule to Construct Consistent Solutions to Stochastic Inverse Problems,'' SIAM Journal on Scientific
# Computing, vol. 40, no. 2, pp. A984–A1011, Jan. 2018.


# Dataset:
# - QoI samples are "tagged" with the key qq and are a 10000×3 array,
#   but we only consider the first QoI samples defined by the first column of this array.
# - Parameter samples are "tagged" with the key pp and are a 10000×100 array.


def load_data():
    filepath = os.path.abspath(os.path.dirname(__file__))
    data_set = sio.loadmat(filepath + '/elliptic_kde100_10K.mat')
    qvals = data_set['qq']  # QoI samples, shape (10k, 3)
    qvals = qvals[:, 0]  # Only using first QoI here --> (10k, 1)
    qvals = np.reshape(qvals, (len(qvals), 1))
    lam = data_set['pp']  # Random parameter samples
    lam = np.transpose(lam)  # reshape to (10k, 100)

    return lam, qvals


def get_prior_samples():
    lam, qvals = load_data()
    return lam


def construct_lowfi_model(X_train, y_train):

    # Fit a simple GP as a surrogate to the deterministic forward problem
    kernel = Matern()
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(X_train, y_train)

    return gp


def construct_lowfi_model_and_get_samples(X_train, y_train, X_test):

    # Fit a simple GP as a surrogate to the deterministic forward problem
    kernel = Matern()
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(X_train, y_train)

    return gp.predict(X_test)


def get_lowfi_samples(gp, X_test, n_lf, fun=lambda x: x):

    if n_lf > X_test.shape[0]:
        print('Number of low-fidelity samples exceeds available data.')
        exit()

    # Choose the specified number of lowfi samples
    indices = np.random.choice(range(X_test.shape[0]), size=n_lf, replace=False)
    X_test = X_test[indices]

    # Predict the lowfi QoIs
    qvals_lowfi = gp.predict(X_test)

    # Transform the output according to a function fun
    qvals_lowfi = fun(qvals_lowfi)

    return qvals_lowfi, indices


def get_highfi_samples(y, indices):

    # Choose hifi samples according to the specified indices
    qvals_highfi = y[indices]

    return qvals_highfi


def find_xy_pair(x, X, Y):

    idx = np.where((X == x).all(axis=1))[0][0]

    return Y[idx, :]


# Exemplary usage
if __name__ == '__main__':

    # Load data
    lam, qvals = load_data()

    # Split data to construct low-fi model
    split = 0.1
    lam_train = lam[0:round(split*lam.shape[0])]
    qvals_train = qvals[0:round(split*lam.shape[0])]
    lam = lam[round(split*lam.shape[0])+1:]
    qvals = qvals[round(split * qvals.shape[0]) + 1:]

    # Construct low-fi model
    lowfi = construct_lowfi_model(lam_train, qvals_train)

    # Get samples for both models
    n = 100
    samples_lowfi, indices = get_lowfi_samples(lowfi, lam, n)
    samples_highfi = get_highfi_samples(qvals, indices)

    # Plot training set
    plt.figure()
    plt.plot(samples_lowfi, samples_highfi, 'k*')
    plt.show()
    exit()