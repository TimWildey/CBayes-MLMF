import numpy as np
from scipy.stats import gaussian_kde as GKDE
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Product, DotProduct
import elliptic_pde
import bmfmc_utils as utils


def create_bmfmc_density(n_hf, fun=lambda x: x):
    #
    # Inputs:
    #   n_hf: number of high-fidelity samples
    #   fun: function that is applied on the low-fidelity outputs (to play with bias)
    #
    # Outputs:
    #   q_hf: the estimated KDE of p(Q)
    #   samples_hf: the samples Q^(i) that were used to estimate p(Q)
    #   lam_lf: the corresponding lambda^(i) which were also used to estimate p(q)
    #

    # Load dataset
    lam, qvals = elliptic_pde.load_data()

    # Split the data for surrogate model creation
    split = 0.05
    lam_train = lam[0:round(split * lam.shape[0])]
    qvals_train = qvals[0:round(split * lam.shape[0])]

    # Construct low-fi model
    lf = elliptic_pde.construct_lowfi_model(lam_train, qvals_train)

    # Remaining data
    lam = lam[round(split * lam.shape[0]) + 1:]
    qvals = qvals[round(split * qvals.shape[0]) + 1:]

    # Remaining number of samples for the low-fidelity model
    n_lf = round((1 - split) * len(qvals)) - 1

    # Sample from the low-fidelity and do KDE to get p(q)
    samples_lf, indices = elliptic_pde.get_lowfi_samples(lf, lam, n_lf, fun)
    lam_lf = lam[indices]
    q_lf = GKDE(samples_lf)

    # Plot the low-fidelity density p(q)
    utils.plot_1d_hist(samples_lf, num=1)
    utils.plot_1d_kde(q_lf, np.min(samples_lf), np.max(samples_lf), linestyle='--', color='k',
                      num=1, label='$p(q)$', title='Low-fidelity density', xlabel="$q$", ylabel="$p(q)$")
    plt.gcf().savefig('../pngout/elliptic_pde_bmfmc_fig1.png', dpi=300)

    # Select high-fidelity model evaluation points
    # 1) Create a uniform grid across the support of p(q)
    x_train_linspace = np.linspace(samples_lf.min(), samples_lf.max(), num=n_hf).reshape(-1, 1)

    # 2) Find the low-fidelity samples closest to the grid points, get the corresponding lambdas
    #    and the corresponding Q^(i)'s
    y_train = np.zeros((n_hf,))
    x_train = np.zeros((n_hf,))
    for i in range(n_hf):
        idx = (np.abs(samples_lf - x_train_linspace[i])).argmin()
        x_train[i] = samples_lf[idx]
        y_train[i] = elliptic_pde.get_highfi_samples(qvals, indices[idx])

    x_train = x_train.reshape(-1, 1)

    # Determine some bounds for plotting
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    y_min = np.min(y_train)
    y_max = np.max(y_train)

    # Plot training data
    utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k', num=2, label='Training',
                       xlim=[x_min, x_max], ylim=[y_min, y_max], title='Training set', xlabel="$q$", ylabel="$Q$",)
    plt.gcf().savefig('../pngout/elliptic_pde_bmfmc_fig2.png', dpi=300)

    # Fit a GP regression model to approximate p(Q|q)
    x_pred = samples_lf.reshape(-1, 1)
    kernel = Product(RBF(), ConstantKernel()) + WhiteKernel() + ConstantKernel() + DotProduct()
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(x_train, y_train)

    # Predict Q|q at all low-fidelity samples
    mu, sigma = gp.predict(x_pred, return_std=True)

    # Generate high-fidelity samples from the predictions
    samples_hf = np.zeros((mu.shape[0],))
    for i in range(mu.shape[0]):
        samples_hf[i] = mu[i] + sigma[i] * np.random.randn()

    # Sort to be able to use the plt.fill
    sort_indices = np.argsort(x_pred, axis=0)
    x_pred = x_pred[sort_indices].reshape(-1, 1)
    y_pred = mu[sort_indices]

    # Plot the regression model
    utils.plot_1d_conf(x_pred, y_pred, sigma[sort_indices], num=3)
    utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k', num=3, label='Training',
                       xlim=[x_min, x_max], ylim=[y_min, y_max], title='Regression model', xlabel="$q$", ylabel="$Q$",)
    plt.gcf().savefig('../pngout/elliptic_pde_bmfmc_fig3.png', dpi=300)

    # Plot samples from the joint density p(q,Q)
    utils.plot_1d_data(x_pred, samples_hf[sort_indices], marker='o', linestyle='', markersize=3, color='C0',
                       xlim=[x_min, x_max], ylim=[y_min, y_max], num=4, label='$p(q,Q)$ samples')
    utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k', xlim=[x_min, x_max],
                       ylim=[y_min, y_max], xlabel="$q$", ylabel="$Q$",
                       num=4, label='Training data', title='Samples from the joint $p(q,Q)$')
    plt.gcf().savefig('../pngout/elliptic_pde_bmfmc_fig4.png', dpi=300)

    # Visualize the approximate joint density p(q,Q)
    x = samples_lf
    y = samples_hf
    xy = np.vstack([x, y])
    z = GKDE(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    utils.plot_2d_scatter(x, y, z, num=5, xlim=[x_min, x_max], ylim=[y_min, y_max],
                          title = 'Approximate joint $p(q,Q)$', xlabel="$q$", ylabel="$Q$")
    plt.gcf().savefig('../pngout/elliptic_pde_bmfmc_fig5.png', dpi=300)

    # Visualize the the exact MC joint density p(q,Q)
    samples_hf_mc = qvals[indices]
    x = samples_lf
    y = samples_hf_mc
    xy = np.vstack([x, y])
    z = GKDE(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    utils.plot_2d_scatter(x, y, z, num=6, xlim=[x_min, x_max], ylim=[y_min, y_max], title='MC joint $p(q,Q)$',
                          xlabel="$q$", ylabel="$Q$")
    plt.gcf().savefig('../pngout/elliptic_pde_bmfmc_fig6.png', dpi=300)

    # Estimate p(Q) using KDE
    q_hf = GKDE(np.ravel(samples_hf))

    # Plot the densities
    q_hf_mc = GKDE(samples_hf_mc)
    utils.plot_1d_kde(q_hf, np.min([x_min, y_min]), np.max([x_max, y_max]), linestyle='-', color='C3', num=7, label='Approximate $p(Q)$')
    utils.plot_1d_kde(q_hf_mc, np.min([x_min, y_min]), np.max([x_max, y_max]), linestyle='--', color='C0', num=7, label='MC reference $p(Q)$')
    utils.plot_1d_kde(q_lf, np.min([x_min, y_min]), np.max([x_max, y_max]), linestyle='--', color='k', xlabel='$q$ / $Q$', ylabel='$p(q)$ / $p(Q)$',
                      num=7, label='Approximate $p(q)$', title='KDE densities')
    plt.gcf().savefig('../pngout/elliptic_pde_bmfmc_fig7.png', dpi=300)

    # Print some statistics
    utils.print_bmfmc_stats(samples_hf_mc, samples_hf, samples_lf)

    # Approximate the CDF (this is rather inefficient)
    y_range = np.linspace(0.6, 0.8, 30)
    cdf_samples = np.zeros((n_lf, y_range.shape[0]))
    for i in range(y_range.shape[0]):
        for j in range(n_lf):
            cdf_samples[j, i] = stats.norm.cdf((y_range[i] - mu[j]) / sigma[j])

    cdf_mean = np.mean(cdf_samples, 0)

    # Calculate the MC CDF
    cdf_mc = np.zeros((y_range.shape[0],))
    for i in range(y_range.shape[0]):
        for j in range(n_lf):
            cdf_mc[i] = cdf_mc[i] + 1 / n_lf * (samples_hf_mc[j] <= y_range[i])

    # Plot the CDFs
    utils.plot_1d_data(y_range, cdf_mean, linestyle='-', color='C3',
                       num=8, label='Approximate mean')
    utils.plot_1d_data(y_range, cdf_mc, linestyle='--', color='C0', xlabel='$Q_0$', ylabel='Pr$[Q \leq Q_0]$',
                       num=8, label='MC mean', title='Estimated CDF')
    plt.gcf().savefig('../pngout/elliptic_pde_bmfmc_fig8.png', dpi=300)

    # Return outputs
    return q_hf, samples_hf, lam_lf


# Exemplary usage
if __name__ == '__main__':

    create_bmfmc_density(n_hf=50, fun=lambda x: np.sin(x))
    # plt.show()
    exit()
