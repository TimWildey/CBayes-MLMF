import numpy as np
from scipy.stats import gaussian_kde as gkde
import matplotlib.pyplot as plt
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Product, DotProduct
import ode_pp
import bmfmc_utils as utils


def create_bmfmc_density(n_hf=20, n_lf=10000):
    #
    # Inputs:
    #   n_hf: number of high-fidelity samples
    #   n_lf: number of low-fidelity samples
    #
    # Outputs:
    #   q_hf: the estimated KDE of p(Q)
    #   samples_hf: the samples Q^(i) that were used to estimate p(Q)
    #   lam_lf: the corresponding lambda^(i) which were also used to estimate p(q)
    #

    # General settings
    n_random = 4
    u0 = np.array([5, 1])
    finalt = 5.0
    dt_hf = 0.01
    dt_lf = 1.0

    # Create settings for the 3 models
    hf_settings = ode_pp.Settings(finalt, dt_hf, u0)
    lf_settings = ode_pp.Settings(finalt, dt_lf, u0)

    # Generate samples from the random variables
    lam_lf = np.random.uniform(low=0.4, high=0.6, size=(n_lf, n_random))

    # Evaluate the lowest-fidelity model i.e. generate samples and apply KDE to get p(q)
    samples_lf = ode_pp.get_qoi_samples(lam_lf, lf_settings)
    q_lf = gkde(samples_lf)

    # Plot the lowest-fidelity density p(q)
    utils.plot_1d_hist(samples_lf, num=1)
    utils.plot_1d_kde(q_lf, np.min(samples_lf), np.max(samples_lf), linestyle='--', color='k',
                      num=1, label='$p(q)$', title='Low-fidelity density', xlabel="$q$", ylabel="$p(q)$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_1lf_fig1.png', dpi=300)

    # Select low-fidelity model evaluation points
    # 1) Create a uniform grid across the support of p(q)
    x_train_linspace = np.linspace(samples_lf.min(), samples_lf.max(), num=n_hf).reshape(-1, 1)

    # 2) Find the low-fidelity samples closest to the grid points and get the corresponding lambdas
    x_train = np.zeros((n_hf,))
    lam_hf = np.zeros((n_hf, n_random))
    for i in range(n_hf):
        idx = (np.abs(samples_lf - x_train_linspace[i])).argmin()
        x_train[i] = samples_lf[idx]
        lam_hf[i, :] = lam_lf[idx, :]

    x_train = x_train.reshape(-1, 1)

    # 3) Evaluate the high-fidelity model at those points
    y_train = ode_pp.get_qoi_samples(lam_hf, hf_settings)

    # Determine some bounds for plotting
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    y_min = np.min(y_train)
    y_max = np.max(y_train)

    # Plot training data
    utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k', num=2, label='Training',
                       xlim=[x_min, x_max], ylim=[y_min, y_max], title='Training set', xlabel="$q$", ylabel="$Q$",)
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_1lf_fig2.png', dpi=300)

    # Fit a GP regression model to approximate p(Q|q)
    x_pred = samples_lf.reshape(-1, 1)
    kernel = Product(RBF(), ConstantKernel()) + WhiteKernel() + ConstantKernel() + DotProduct()
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(x_train, y_train)

    # Predict Q|q at all low-fidelity samples
    mu, sigma = gp.predict(x_pred, return_std=True)

    # Generate low-fidelity samples from the predictions
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
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_1lf_fig3.png', dpi=300)

    # Plot samples from the joint density p(Q|q)
    utils.plot_1d_data(x_pred, samples_hf[sort_indices], marker='o', linestyle='', markersize=3, color='C0',
                       xlim=[x_min, x_max], ylim=[y_min, y_max], num=4, label='$p(q,Q)$ samples')
    utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k', xlim=[x_min, x_max],
                       ylim=[y_min, y_max], xlabel="$q$", ylabel="$Q$",
                       num=4, label='Training data', title='Samples from the joint $p(q,Q)$')
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_1lf_fig4.png', dpi=300)

    # Visualize the approximate joint density p(Q|q)
    utils.plot_2d_scatter(samples_x=samples_lf, samples_y=samples_hf, num=5, xlim=[x_min, x_max],
                          ylim=[y_min, y_max], title='Approximate joint $p(q,Q)$', xlabel="$q$", ylabel="$Q$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_1lf_fig5.png', dpi=300)

    # Visualize the approximate joint density p(Q|q) in a contour plot
    utils.plot_2d_contour(samples_x=samples_lf, samples_y=samples_hf, num=6, xlim=[x_min, x_max],
                          ylim=[y_min, y_max], title='Approximate joint $p(q,Q)$', xlabel="$q$", ylabel="$Q$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_1lf_fig6.png', dpi=300)

    # Visualize the exact MC joint density p(Q|q)
    samples_hf_mc = ode_pp.get_qoi_samples(lam_lf, hf_settings)
    utils.plot_2d_scatter(samples_x=samples_lf, samples_y=samples_hf_mc, num=7, xlim=[x_min, x_max],
                          ylim=[y_min, y_max], title='MC joint $p(q,Q)$', xlabel="$q$", ylabel="$Q$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_1lf_fig7.png', dpi=300)

    # Visualize the the exact MC joint density pp(Q|q) in a contour plot
    utils.plot_2d_contour(samples_x=samples_lf, samples_y=samples_hf_mc, num=8, xlim=[x_min, x_max],
                          ylim=[y_min, y_max], title='Approximate joint $p(q,Q)$', xlabel="$q$", ylabel="$Q$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_1lf_fig8.png', dpi=300)

    # Estimate p(Q) using KDE
    q_hf = gkde(np.ravel(samples_hf))

    # Plot the densities
    samples_hf_mc = ode_pp.get_qoi_samples(lam_lf, hf_settings)
    q_hf_mc = gkde(samples_hf_mc)
    utils.plot_1d_kde(q_hf, np.min([x_min, y_min, np.min(samples_hf_mc)]),
                      np.max([x_max, y_max, np.max(samples_hf_mc)]),
                      linestyle='-', color='C3', num=9, label='Approximate $p(Q)$')
    utils.plot_1d_kde(q_hf_mc, np.min([x_min, y_min, np.min(y_train)]), np.max([x_max, y_max, np.max(y_train)]),
                      linestyle='--', color='C0', num=9, label='MC reference $p(Q)$')
    utils.plot_1d_kde(q_lf, np.min([x_min, y_min, np.min(y_train)]), np.max([x_max, y_max, np.max(y_train)]),
                      linestyle='--', color='g', xlabel='$q$ / $Q$', ylabel='$p(q)$ / $p(Q)$',
                      num=9, label='Approximate $p(q)$', title='KDE densities')
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_1lf_fig9.png', dpi=300)

    # Print some statistics
    utils.print_bmfmc_stats(samples_hf_mc, samples_hf, samples_lf)

    # Return outputs
    return q_hf, samples_hf, lam_lf


# Exemplary usage
if __name__ == '__main__':

    create_bmfmc_density(n_hf=10, n_lf=10000)
    # plt.show()
    exit()
