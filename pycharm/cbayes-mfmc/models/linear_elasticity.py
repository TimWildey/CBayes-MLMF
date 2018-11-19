import os
import numpy as np
import matplotlib.pyplot as plt

# A linear elasticity example

# Dataset:
# - low-fidelity: sample_data_c_10K
# - high-fidelity: sample_data_f_100
# - high-fidelity: sample_data_f_100_ce

# QoI:
# - The average y-displacement in upper half of connector


def load_data():
    filepath = os.path.abspath(os.path.dirname(__file__))

    lf_data = np.loadtxt(filepath + '/sample_data_c_10K.dat')
    # hf_data = np.loadtxt(filepath + '/sample_data_f_100.dat')
    hf_data = np.loadtxt(filepath + '/sample_data_f_100_ce.dat')

    return lf_data*1000, hf_data*1000


# Exploring the data
if __name__ == '__main__':

    # Load data
    lf, hf = load_data()

    plt.figure()
    plt.plot(lf[:100], hf, 'k*')
    plt.show()

    print('LF-HF correlations: %f' % np.corrcoef(lf[:100], hf)[0, 1])
    exit()
