# Standard stuff
import numpy as np
import matplotlib.pyplot as plt

# Model stuff
import elliptic_pde

# Framework stuff
from Distribution import Distribution
from Model import Model
from MFMC import MFMC

# ------------------------------------------------- Config - General -------- #

np.random.seed(42)

# Number of samples for the Monte Carlo reference
n_mc_ref = int(1e4)

# ----------------------------------------------------Config - MFMC -------- #

# Training set selection strategy
training_set_strategy = 'support_covering'

# Regression model type
regression_type = 'heteroscedastic_gaussian_process'

# ------------------------------------------------------------- Main -------- #

if __name__ == '__main__':

    # No. of averaging runs
    n_avg = 100

    # Evaluations
    n_grid = 10

    # Monte Carlo
    n_evals_mc = np.logspace(np.log10(30), np.log10(1000), n_grid)
    n_evals_mc = np.round(n_evals_mc).astype(int)

    # MFMC 1lf
    n_evals_mfmc_hf_1lf = np.logspace(np.log10(8), np.log10(150), n_grid)
    n_evals_mfmc_hf_1lf = np.round(n_evals_mfmc_hf_1lf).astype(int)
    n_evals_mfmc_lf_1lf = np.logspace(np.log10(150), np.log10(10000), n_grid)
    # n_evals_mfmc_lf_1lf = n_grid * [10000]
    n_evals_mfmc_lf_1lf = np.round(n_evals_mfmc_lf_1lf).astype(int)

    # MFMC 2lf
    n_evals_mfmc_hf_2lf = np.logspace(np.log10(5), np.log10(20), n_grid)
    n_evals_mfmc_hf_2lf = np.round(n_evals_mfmc_hf_2lf).astype(int)
    n_evals_mfmc_mf_2lf = np.logspace(np.log10(10), np.log10(150), n_grid)
    n_evals_mfmc_mf_2lf = np.round(n_evals_mfmc_mf_2lf).astype(int)
    n_evals_mfmc_lf_2lf = np.logspace(np.log10(150), np.log10(10000), n_grid)
    # n_evals_mfmc_lf_2lf = n_grid * [10000]
    n_evals_mfmc_lf_2lf = np.round(n_evals_mfmc_lf_2lf).astype(int)

    # Costs
    costs_hf = 1.0
    costs_mf = 0.25 * costs_hf
    costs_lf = 0.25 * costs_mf
    total_costs_1lf = costs_hf * n_evals_mfmc_hf_1lf + costs_lf * n_evals_mfmc_lf_1lf
    total_costs_1lf = np.round(total_costs_1lf).astype(int)
    total_costs_2lf = (costs_hf + costs_mf) * n_evals_mfmc_hf_2lf + \
        costs_mf * n_evals_mfmc_mf_2lf + costs_lf * n_evals_mfmc_lf_2lf
    total_costs_2lf = np.round(total_costs_2lf).astype(int)

    # Load data
    n_qoi = 1
    prior_pf_samples = elliptic_pde.load_data()
    prior_pf_samples_hf = prior_pf_samples[-1][:, 0:n_qoi]
    prior_pf_samples_mf = prior_pf_samples[1][:, 0:n_qoi] ** 1.1
    prior_pf_samples_lf = prior_pf_samples[0][:, 0:n_qoi] ** 1.2
    prior_samples = np.reshape(range(n_mc_ref), (n_mc_ref, 1))  # we only need some id here

    # Create the MC reference samples
    ref_prior_pf_samples = prior_pf_samples_hf[:n_mc_ref]

    # Prior
    p_prior = Distribution(prior_samples, rv_name='$\lambda$', label='Prior', kde=False)

    # Prior push-forward
    ref_p_prior_pf = Distribution(ref_prior_pf_samples, rv_name='$Q$', label='Prior-PF')
    ref_p_prior_pf.eval_kernel_density()

    # Observed density
    obs_loc = [0.71]
    obs_scale = [0.02]
    obs_samples = np.random.randn(n_mc_ref, len(obs_scale)) * obs_scale + obs_loc
    obs_samples = np.reshape(obs_samples, (n_mc_ref, np.shape(obs_samples)[1]))
    p_obs = Distribution(obs_samples, rv_name='$Q$', label='Observed')
    p_obs_evals = p_obs.kernel_density(ref_prior_pf_samples.T)

    # Reference r
    ref_r = p_obs_evals / (ref_p_prior_pf.kernel_density(ref_prior_pf_samples.T) + 1.0e-10)

    l1_posterior_1hf_avg = np.zeros((n_grid, ))
    l1_posterior_1hf_1lf_avg = np.zeros((n_grid, ))
    l1_posterior_1hf_2lf_avg = np.zeros((n_grid, ))
    for k in range(n_avg):
        print('\nRun %d / %d' % (k+1, n_avg))

        # -------------- 1 HF

        l1_posterior_1hf = []
        for idx, n_evals in enumerate(n_evals_mc):
            indices = np.random.choice(range(prior_pf_samples_hf.shape[0]), size=n_evals, replace=False)

            # Monte Carlo model
            prior_pf_samples = prior_pf_samples_hf[indices]
            p_prior_pf = Distribution(prior_pf_samples, rv_name='$Q$', label='Prior-PF')

            # l1 error between mfmc posterior and reference posterior
            r = p_obs_evals / (p_prior_pf.kernel_density(ref_prior_pf_samples.T) + 1.0e-10)
            l1_posterior_1hf.append(np.mean(np.abs(ref_r - r)))

        # -------------- 1 HF, 1 LF

        l1_posterior_1hf_1lf = []
        for idx, n_evals in enumerate(n_evals_mfmc_hf_1lf):
            n_evals = [n_evals_mfmc_lf_1lf[idx], n_evals]
            indices = np.random.choice(range(prior_samples.shape[0]), size=n_evals[0], replace=False)

            # Create a low-fidelity model
            lf_model = Model(
                eval_fun=lambda x, samples=prior_pf_samples_lf: elliptic_pde.find_xy_pair(x, prior_samples, samples),
                rv_samples=prior_samples[indices], rv_samples_pred=prior_samples[indices], n_evals=n_evals[0],
                n_qoi=n_qoi, rv_name='$q_0$', label='Low-fidelity')

            # Create a high-fidelity model
            hf_model = Model(
                eval_fun=lambda x, samples=prior_pf_samples_hf: elliptic_pde.find_xy_pair(x, prior_samples, samples),
                n_evals=n_evals[-1], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            models = [lf_model, hf_model]

            # Setup MFMC
            mfmc = MFMC(models=models, training_set_strategy=training_set_strategy, regression_type=regression_type)

            # Apply MFMC
            mfmc.apply_mfmc_framework(verbose=False)
            prior_pf_samples = mfmc.get_samples()[-1, :, :]
            p_prior_pf = Distribution(prior_pf_samples, rv_name='$Q$', label='Prior-PF')

            # l1 error between mfmc posterior and reference posterior
            r = p_obs_evals / (p_prior_pf.kernel_density(ref_prior_pf_samples.T) + 1.0e-10)
            l1_posterior_1hf_1lf.append(np.mean(np.abs(ref_r - r)))

        # -------------- 1 HF, 2 LF

        l1_posterior_1hf_2lf = []
        for idx, n_evals in enumerate(n_evals_mfmc_hf_2lf):
            n_evals = [n_evals_mfmc_lf_2lf[idx], n_evals_mfmc_mf_2lf[idx], n_evals]
            indices = np.random.choice(range(prior_samples.shape[0]), size=n_evals[0], replace=False)

            # Create a low-fidelity model
            lf_model = Model(
                eval_fun=lambda x, samples=prior_pf_samples_lf: elliptic_pde.find_xy_pair(x, prior_samples, samples),
                rv_samples=prior_samples[indices], rv_samples_pred=prior_samples[indices], n_evals=n_evals[0],
                n_qoi=n_qoi, rv_name='$q_0$', label='Low-fidelity')

            # Create a mid-fidelity model
            mf_model = Model(
                eval_fun=lambda x, samples=prior_pf_samples_mf: elliptic_pde.find_xy_pair(x, prior_samples, samples),
                n_evals=n_evals[1], n_qoi=n_qoi, rv_name='$q_1$', label='Mid-fidelity')

            # Create a high-fidelity model
            hf_model = Model(
                eval_fun=lambda x, samples=prior_pf_samples_hf: elliptic_pde.find_xy_pair(x, prior_samples, samples),
                n_evals=n_evals[-1], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            models = [lf_model, mf_model, hf_model]

            # Setup MFMC
            mfmc = MFMC(models=models, training_set_strategy=training_set_strategy, regression_type=regression_type)

            # Apply MFMC
            mfmc.apply_mfmc_framework(verbose=False)
            prior_pf_samples = mfmc.get_samples()[-1, :, :]
            p_prior_pf = Distribution(prior_pf_samples, rv_name='$Q$', label='Prior-PF')

            # l1 error between mfmc posterior and reference posterior
            r = p_obs_evals / (p_prior_pf.kernel_density(ref_prior_pf_samples.T) + 1.0e-10)
            l1_posterior_1hf_2lf.append(np.mean(np.abs(ref_r - r)))

        l1_posterior_1hf_avg += 1 / n_avg * np.asarray(l1_posterior_1hf)
        l1_posterior_1hf_1lf_avg += 1 / n_avg * np.asarray(l1_posterior_1hf_1lf)
        l1_posterior_1hf_2lf_avg += 1 / n_avg * np.asarray(l1_posterior_1hf_2lf)

    n_evals_all = np.logspace(np.log10(15), np.log10(1000), 2)
    n_evals_all = np.round(n_evals_all).astype(int)

    plt.figure()
    plt.semilogx(np.squeeze(n_evals_mc), l1_posterior_1hf_avg, '-o', label='1 HF')
    plt.semilogx(np.squeeze(total_costs_1lf), l1_posterior_1hf_1lf_avg, '-o', label='1 HF, 1 LF')
    plt.semilogx(np.squeeze(total_costs_2lf), l1_posterior_1hf_2lf_avg, '-o', label='1 HF, 1 MF, 1 LF')
    plt.semilogx(np.squeeze(n_evals_all), np.zeros((len(n_evals_all))), 'k--')
    plt.xlabel('$C_{\mathrm{tot}}$')
    plt.ylabel('L1 error')
    plt.legend(loc='upper right')
    plt.grid(b=True)
    plt.gcf().savefig('elliptic_pde_1qoi_posterior_convergence_l1.eps', dpi=300)
    np.save('elliptic_pde_1qoi_posterior_convergence_l1', [l1_posterior_1hf_avg,
                                                           l1_posterior_1hf_1lf_avg,
                                                           l1_posterior_1hf_2lf_avg])

# --------------------------------------------------------------------------- #
