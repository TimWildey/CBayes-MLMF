import numpy as np
from scipy.stats import gaussian_kde as gkde
import matplotlib.pyplot as plt
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Product, DotProduct
import ode_pp
import bmfmc_utils as utils


def create_bmfmc_density(n_hf=20, n_lf_1=100, n_lf_2=10000):
    #
    # Inputs:
    #   n_hf: number of high-fidelity samples
    #   n_lf_1: number of first hierarchy low-fidelity samples (low-fidelity)
    #   n_lf_2: number of second hierarchy low-fidelity samples (lowest-fidelity)
    #
    # Outputs:
    #   q_hf: the estimated KDE of p(Q)
    #   samples_hf: the samples Q^(i) that were used to estimate p(Q)
    #   lam_lf_2: the corresponding lambda^(i) which were also used to estimate p(q_1)
    #

    # General settings
    n_random = 4
    u0 = np.array([5, 1])
    finalt = 1.0
    dt_hf = 0.01
    dt_lf_1 = 0.7
    dt_lf_2 = 0.9

    # Create settings for the 3 models
    hf_settings = ode_pp.Settings(finalt, dt_hf, u0)
    lf_1_settings = ode_pp.Settings(finalt, dt_lf_1, u0)
    lf_2_settings = ode_pp.Settings(finalt, dt_lf_2, u0)

    # Generate samples from the random variables
    lam_lf_2 = np.random.uniform(low=0.4, high=0.6, size=(n_lf_2, n_random))

    # Evaluate the lowest-fidelity model i.e. generate samples and apply KDE to get p(q_2)
    samples_lf_2 = ode_pp.get_qoi_samples(lam_lf_2, lf_2_settings)
    q_lf_2 = gkde(samples_lf_2)

    # Plot the lowest-fidelity density p(q)
    utils.plot_1d_hist(samples_lf_2, num=1)
    utils.plot_1d_kde(q_lf_2, np.min(samples_lf_2), np.max(samples_lf_2), linestyle='--', color='k',
                      num=1, label='$p(q)$', title='Low-fidelity density', xlabel="$q$", ylabel="$p(q)$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig1.png', dpi=300)

    # Select low-fidelity model evaluation points
    # 1) Create a uniform grid across the support of p(q_2)
    x_train_linspace = np.linspace(samples_lf_2.min(), samples_lf_2.max(), num=n_lf_1).reshape(-1, 1)

    # 2) Find the low-fidelity samples closest to the grid points and get the corresponding lambdas
    x_train = np.zeros((n_lf_1,))
    lam_lf_1 = np.zeros((n_lf_1, n_random))
    for i in range(n_lf_1):
        idx = (np.abs(samples_lf_2 - x_train_linspace[i])).argmin()
        x_train[i] = samples_lf_2[idx]
        lam_lf_1[i, :] = lam_lf_2[idx, :]

    x_train = x_train.reshape(-1, 1)

    # 3) Evaluate the high-fidelity model at those points
    y_train = ode_pp.get_qoi_samples(lam_lf_1, lf_1_settings)

    # Determine some bounds for plotting
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    y_min = np.min(y_train)
    y_max = np.max(y_train)

    # Plot training data
    utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k', num=2, label='Training',
                       xlim=[x_min, x_max], ylim=[y_min, y_max], title='Training set', xlabel="$q$", ylabel="$Q$",)
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig2.png', dpi=300)

    # Fit a GP regression model to approximate p(q_1|q_2)
    x_pred = samples_lf_2.reshape(-1, 1)
    kernel = Product(RBF(), ConstantKernel()) + WhiteKernel() + ConstantKernel() + DotProduct()
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(x_train, y_train)

    # Predict q_1|q_2 at all low-fidelity samples
    mu, sigma = gp.predict(x_pred, return_std=True)

    # Generate low-fidelity samples from the predictions
    samples_lf_1 = np.zeros((mu.shape[0],))
    for i in range(mu.shape[0]):
        samples_lf_1[i] = mu[i] + sigma[i] * np.random.randn()

    # Sort to be able to use the plt.fill
    sort_indices = np.argsort(x_pred, axis=0)
    x_pred = x_pred[sort_indices].reshape(-1, 1)
    y_pred = mu[sort_indices]

    # Plot the regression model
    utils.plot_1d_conf(x_pred, y_pred, sigma[sort_indices], num=3)
    utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k', num=3, label='Training',
                       xlim=[x_min, x_max], ylim=[y_min, y_max], title='Regression model', xlabel="$q_2$", ylabel="$q_1$",)
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig3.png', dpi=300)

    # Plot samples from the joint density p(q_1|q_2)
    utils.plot_1d_data(x_pred, samples_lf_1[sort_indices], marker='o', linestyle='', markersize=3, color='C0',
                       xlim=[x_min, x_max], ylim=[y_min, y_max], num=4, label='$p(q_2,q_1)$ samples')
    utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k', xlim=[x_min, x_max],
                       ylim=[y_min, y_max], xlabel="$q_2$", ylabel="$q_1$",
                       num=4, label='Training data', title='Samples from the joint $p(q_2,q_1)$')
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig4.png', dpi=300)

    # Visualize the approximate joint density p(q_1|q_2)
    utils.plot_2d_scatter(samples_x=samples_lf_2, samples_y=samples_lf_1, num=5, xlim=[x_min, x_max],
                          ylim=[y_min, y_max], title='Approximate joint $p(q_2,q_1)$', xlabel="$q_2$", ylabel="$q_1$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig5.png', dpi=300)

    # Visualize the approximate joint density p(q_1|q_2) in a contour plot
    utils.plot_2d_contour(samples_x=samples_lf_2, samples_y=samples_lf_1, num=6, xlim=[x_min, x_max],
                          ylim=[y_min, y_max], title='Approximate joint $p(q_2,q_1)$', xlabel="$q_2$", ylabel="$q_1$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig6.png', dpi=300)

    # Visualize the exact MC joint density p(q_1|q_2)
    samples_lf_1_mc = ode_pp.get_qoi_samples(lam_lf_2, lf_1_settings)
    utils.plot_2d_scatter(samples_x=samples_lf_2, samples_y=samples_lf_1_mc, num=7, xlim=[x_min, x_max],
                          ylim=[y_min, y_max], title='MC joint $p(q_2,q_1)$', xlabel="$q_2$", ylabel="$q_1$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig7.png', dpi=300)

    # Visualize the the exact MC joint density pp(q_1|q_2) in a contour plot
    utils.plot_2d_contour(samples_x=samples_lf_2, samples_y=samples_lf_1_mc, num=8, xlim=[x_min, x_max],
                          ylim=[y_min, y_max], title='Approximate joint $p(q_2,q_1)$', xlabel="$q_2$", ylabel="$q_1$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig8.png', dpi=300)

    # Estimate p(q_1) using KDE
    q_lf_1 = gkde(np.ravel(samples_lf_1))

    # Select high-fidelity model evaluation points
    # 1) Create a uniform grid across the support of p(q_1)
    x_train_linspace = np.linspace(samples_lf_1.min(), samples_lf_1.max(), num=n_hf).reshape(-1, 1)

    # 2) Find the low-fidelity samples closest to the grid points and get the corresponding lambdas
    # x_train = np.zeros((n_hf, 2))
    x_train = np.zeros((n_hf, 1))
    lam_hf = np.zeros((n_hf, n_random))
    for i in range(n_hf):
        idx = (np.abs(samples_lf_1 - x_train_linspace[i])).argmin()
        # x_train[i, :] = np.array(samples_lf_2[idx], samples_lf_1[idx])
        x_train[i] = samples_lf_1[idx].reshape(-1, 1)
        lam_hf[i, :] = lam_lf_2[idx, :]

    # 3) Evaluate the high-fidelity model at those points
    y_train = ode_pp.get_qoi_samples(lam_hf, hf_settings)

    # Determine some bounds for plotting
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    y_min = np.min(y_train)
    y_max = np.max(y_train)

    # Plot training data
    utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k', num=9, label='Training',
                       xlim=[x_min, x_max], ylim=[y_min, y_max], title='Training set', xlabel="$q$", ylabel="$Q$", )
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig9.png', dpi=300)

    # Fit a GP regression model to approximate p(Q|q)
    # x_pred = np.vstack([samples_lf_2, samples_lf_1])
    # x_pred = np.transpose(x_pred)
    x_pred = samples_lf_1.reshape(-1, 1)
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
    utils.plot_1d_conf(x_pred, y_pred, sigma[sort_indices], num=10)
    utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k', num=10, label='Training',
                       xlim=[x_min, x_max], ylim=[y_min, y_max], title='Regression model', xlabel="$q_1$",
                       ylabel="$Q$", )
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig10.png', dpi=300)

    # Plot samples from the joint density p(Q|q_1)
    utils.plot_1d_data(x_pred, samples_hf[sort_indices], marker='o', linestyle='', markersize=3, color='C0',
                       xlim=[x_min, x_max], ylim=[y_min, y_max], num=11, label='$p(q_1,Q)$ samples')
    utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k', xlim=[x_min, x_max],
                       ylim=[y_min, y_max], xlabel="$q_1$", ylabel="$Q$",
                       num=11, label='Training data', title='Samples from the joint $p(q_1,Q)$')
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig11.png', dpi=300)

    # Visualize the approximate joint density p(Q|q_1)
    utils.plot_2d_scatter(samples_x=samples_lf_1, samples_y=samples_hf, num=12, xlim=[x_min, x_max],
                          ylim=[y_min, y_max], title='Approximate joint $p(q_1,Q)$', xlabel="$q_1$",
                          ylabel="$Q$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig12.png', dpi=300)

    # Visualize the approximate joint density p(Q|q_1) in a contour plot
    utils.plot_2d_contour(samples_x=samples_lf_1, samples_y=samples_hf, num=13, xlim=[x_min, x_max],
                          ylim=[y_min, y_max], title='Approximate joint $p(q_1,Q)$', xlabel="$q_1$",
                          ylabel="$Q$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig13.png', dpi=300)

    # Visualize the exact MC joint density p(Q|q_1)
    samples_hf_mc = ode_pp.get_qoi_samples(lam_lf_2, hf_settings)
    utils.plot_2d_scatter(samples_x=samples_lf_1, samples_y=samples_hf, num=14, xlim=[x_min, x_max],
                          ylim=[y_min, y_max], title='MC joint $p(q_1,Q)$', xlabel="$q_1$", ylabel="$Q$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig14.png', dpi=300)

    # Visualize the the exact MC joint density pp(Q|q_1) in a contour plot
    utils.plot_2d_contour(samples_x=samples_lf_1, samples_y=samples_hf_mc, num=15, xlim=[x_min, x_max],
                          ylim=[y_min, y_max], title='Approximate joint $p(q_1,Q)$', xlabel="$q_1$",
                          ylabel="$Q$")
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig15.png', dpi=300)

    # Estimate p(Q) using KDE
    q_hf = gkde(np.ravel(samples_hf))

    # Plot the densities
    q_hf_mc = gkde(samples_hf_mc)
    utils.plot_1d_kde(q_hf, np.min([x_min, y_min, np.min(samples_hf_mc)]), np.max([x_max, y_max, np.max(samples_hf_mc)]),
                      linestyle='-', color='C3', num=16, label='Approximate $p(Q)$')
    utils.plot_1d_kde(q_hf_mc, np.min([x_min, y_min, np.min(y_train)]), np.max([x_max, y_max, np.max(y_train)]),
                      linestyle='--', color='C0', num=16, label='MC reference $p(Q)$')
    utils.plot_1d_kde(q_lf_1, np.min([x_min, y_min, np.min(y_train)]), np.max([x_max, y_max, np.max(y_train)]),
                      linestyle='--', color='k', num=16, label='Approximate $p(q_1)$', title='KDE densities')
    utils.plot_1d_kde(q_lf_2, np.min([x_min, y_min, np.min(y_train)]), np.max([x_max, y_max, np.max(y_train)]),
                      linestyle='--', color='g', xlabel='$q_1$ / $q_2$ / $Q$', ylabel='$p(q_1)$ / $p(q_2)$ / $p(Q)$',
                      num=16, label='Approximate $p(q_2)$', title='KDE densities')
    plt.gcf().savefig('../pngout/ode_pp_bmfmc_fig16.png', dpi=300)

    # Print some statistics
    utils.print_bmfmc_stats_2lf(samples_hf_mc, samples_hf, samples_lf_1, samples_lf_2)

    # Return outputs
    return q_hf, samples_hf, lam_lf_2


# Exemplary usage
if __name__ == '__main__':

    create_bmfmc_density(n_hf=10, n_lf_1=100, n_lf_2=10000)
    # plt.show()
    exit()
