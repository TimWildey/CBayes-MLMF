import os
import numpy as np
import matplotlib.pyplot as plt

# A linear (crystal) elasticity example

# Dataset:
# - low-fidelity elasticity: sample_data_c_10K
# - high-fidelity elasticity: sample_data_f_100
# - high-fidelity crystal elasticity: sample_data_f_100_ce
# - low-fidelity samples: sample_points_c_10k.dat

# QoI:
# - The average y-displacement in upper half of connector


def load_data():
    filepath = os.path.abspath(os.path.dirname(__file__))

    lf_data = np.loadtxt(filepath + '/sample_data_c_10K.dat')
    hf_data = np.loadtxt(filepath + '/sample_data_f_100_ce.dat')
    # hf_data = np.loadtxt(filepath + '/sample_data_f_100.dat')
    lf_samples = np.loadtxt(filepath + '/sample_points_c_10K.dat')
    
    return 1000*lf_data, 1000*hf_data, lf_samples


# Exploring the data
if __name__ == '__main__':

    # Load data
    lf, hf, sp = load_data()

    plt.figure()
    plt.plot(lf[:100], hf, 'k*')
    plt.show()

    print('LF-HF correlations: %f' % np.corrcoef(lf[:100], hf)[0, 1])
