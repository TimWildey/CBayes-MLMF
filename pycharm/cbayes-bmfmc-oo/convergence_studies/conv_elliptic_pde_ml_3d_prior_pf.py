# Standard stuff
import numpy as np
import matplotlib.pyplot as plt

# Model stuff
import elliptic_pde_ml

# Framework stuff
from Distribution import Distribution
from Model import Model
from BMFMC import BMFMC
from CBayes import CBayesPosterior

# ------------------------------------------------- Config - General -------- #

np.random.seed(42)

# Number of lowest-fidelity samples (1e4 should be fine for 1 QoI)
n_mc_ref = int(5e4)

# ----------------------------------------------------Config - BMFMC -------- #

# Training set selection strategy (support_covering, support_covering_adaptive, sampling, sampling_adaptive)
training_set_strategy = 'sampling'

# Regression model type (gaussian_process, heteroscedastic_gaussian_process, decoupled_heteroscedastic_gaussian_process)
regression_type = 'heteroscedastic_gaussian_process'

# ------------------------------------------------------------- Main -------- #

if __name__ == '__main__':

    # 10, 50 works good
    n_fac_mf = 10
    n_fac_lf = 50

    n_evals_mc = np.logspace(np.log10(15), np.log10(1200), 10)
    n_evals_mc = np.round(n_evals_mc).astype(int)
    n_evals_bmfmc_hf = np.logspace(np.log10(10), np.log10(500), 10)
    n_evals_bmfmc_hf = np.round(n_evals_bmfmc_hf).astype(int)
    n_evals_bmfmc_mf = np.logspace(np.log10(20), np.log10(500), 10)
    n_evals_bmfmc_mf = np.round(n_evals_bmfmc_mf).astype(int)
    n_evals_bmfmc_lf = np.logspace(np.log10(200), np.log10(10000), 10)
    n_evals_bmfmc_lf = np.round(n_evals_bmfmc_lf).astype(int)
    # n_evals_bmfmc_mf = n_fac_mf * n_evals_bmfmc_hf
    # n_evals_bmfmc_lf = n_fac_lf * n_evals_bmfmc_hf
    n_evals_all = n_evals_mc.tolist() + list(set(n_evals_bmfmc_hf.tolist()) - set(n_evals_mc.tolist()))
    n_evals_all.sort()

    costs_hf = 1.0
    costs_mf = 0.1 * costs_hf
    costs_lf = 0.1 * costs_mf
    total_costs_1lf = (costs_hf + costs_lf) * n_evals_bmfmc_hf + costs_lf * n_evals_bmfmc_lf
    total_costs_1lf = np.round(total_costs_1lf).astype(int)
    total_costs_2lf = (costs_hf + costs_mf) * n_evals_bmfmc_hf + (costs_mf + costs_lf) * n_evals_bmfmc_mf + costs_lf * n_evals_bmfmc_lf
    total_costs_2lf = np.round(total_costs_2lf).astype(int)

    # Load data
    n_qoi = 3
    prior_pf_samples = elliptic_pde_ml.load_data(h=40, n_models=3)
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

    # Pre evaluations
    ref_prior_pf_samples = np.squeeze(ref_prior_pf_samples)
    ref_p_prior_pf_evals = ref_p_prior_pf.kernel_density(ref_prior_pf_samples.T)

    # -------------- 1 HF

    kls_prior_pf_1hf = []
    for idx, n_evals in enumerate(n_evals_mc):
        print('\nCalculating MC model %d / %d ...' % (idx + 1, len(n_evals_mc)))

        # Brute force Monte Carlo
        prior_pf_samples = prior_pf_samples_hf[0:n_evals]
        p_prior_pf = Distribution(prior_pf_samples, rv_name='$Q$', label='Prior-PF')

        # kl between prior push-forward and reference push-forward
        kls_prior_pf_1hf.append(ref_p_prior_pf.calculate_kl_divergence(p_prior_pf))

    # -------------- 1 HF, 1 LF

    kls_prior_pf_1hf_1lf = []
    for idx, n_evals in enumerate(n_evals_bmfmc_hf):
        print('\nCalculating BMFMC model %d / %d ...' % (idx + 1, len(n_evals_bmfmc_hf)))
        n_evals = [n_evals_bmfmc_lf[idx], n_evals]

        # Create a low-fidelity model
        lf_model = Model(
            eval_fun=lambda x, samples=prior_pf_samples_lf: elliptic_pde_ml.find_xy_pair(x, prior_samples, samples),
            rv_samples=prior_samples[:n_evals[0]], rv_samples_pred=prior_samples[:n_evals[0]], n_evals=n_evals[0],
            n_qoi=n_qoi, rv_name='$q_0$', label='Low-fidelity')

        # Create a high-fidelity model
        hf_model = Model(
            eval_fun=lambda x, samples=prior_pf_samples_hf: elliptic_pde_ml.find_xy_pair(x, prior_samples, samples),
            n_evals=n_evals[-1], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

        models = [lf_model, hf_model]

        # Setup BMFMC
        bmfmc = BMFMC(models=models,
                      training_set_strategy=training_set_strategy, regression_type=regression_type)

        # Apply BMFMC
        bmfmc.apply_bmfmc_framework()
        prior_pf_samples = bmfmc.get_samples()[-1, :, :]
        p_prior_pf = Distribution(prior_pf_samples, rv_name='$Q$', label='Prior-PF')

        # kl between prior push-forward and reference push-forward
        kls_prior_pf_1hf_1lf.append(ref_p_prior_pf.calculate_kl_divergence(p_prior_pf))

    # -------------- 1 HF, 2 LF

    kls_prior_pf_1hf_2lf = []
    for idx, n_evals in enumerate(n_evals_bmfmc_hf):
        n_evals = [n_evals_bmfmc_lf[idx], n_evals_bmfmc_mf[idx], n_evals]
        print('\nCalculating BMFMC multi-model %d / %d ...' % (idx + 1, len(n_evals_bmfmc_hf)))

        # Create a low-fidelity model
        lf_model = Model(
            eval_fun=lambda x, samples=prior_pf_samples_lf: elliptic_pde_ml.find_xy_pair(x, prior_samples, samples),
            rv_samples=prior_samples[:n_evals[0]], rv_samples_pred=prior_samples[:n_evals[0]], n_evals=n_evals[0],
            n_qoi=n_qoi, rv_name='$q_0$', label='Low-fidelity')

        # Create a mid-fidelity model
        mf_model = Model(
            eval_fun=lambda x, samples=prior_pf_samples_mf: elliptic_pde_ml.find_xy_pair(x, prior_samples, samples),
            n_evals=n_evals[1], n_qoi=n_qoi, rv_name='$q_1$', label='Mid-fidelity')

        # Create a high-fidelity model
        hf_model = Model(
            eval_fun=lambda x, samples=prior_pf_samples_hf: elliptic_pde_ml.find_xy_pair(x, prior_samples, samples),
            n_evals=n_evals[-1], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

        models = [lf_model, mf_model, hf_model]

        # Setup BMFMC
        bmfmc = BMFMC(models=models,
                      training_set_strategy=training_set_strategy, regression_type=regression_type)

        # Apply BMFMC
        bmfmc.apply_bmfmc_framework()
        prior_pf_samples = bmfmc.get_samples()[-1, :, :]
        p_prior_pf = Distribution(prior_pf_samples, rv_name='$Q$', label='Prior-PF')

        # kl between prior push-forward and reference push-forward
        kls_prior_pf_1hf_2lf.append(ref_p_prior_pf.calculate_kl_divergence(p_prior_pf))

    plt.figure()
    plt.semilogx(np.squeeze(n_evals_mc), kls_prior_pf_1hf, '-o', label='1 HF')
    plt.semilogx(np.squeeze(total_costs_1lf), kls_prior_pf_1hf_1lf, '-o', label='1 HF, 1 LF')
    plt.semilogx(np.squeeze(total_costs_2lf), kls_prior_pf_1hf_2lf, '-o', label='1 HF, 1 MF, 1 LF')
    plt.semilogx(np.squeeze(n_evals_all), np.zeros((len(n_evals_all))), 'k--')
    plt.xlabel('Costs')
    plt.ylabel('KL')
    plt.legend(loc='upper right')
    plt.grid(b=True)
    plt.gcf().savefig('elliptic_pde_ml_3qoi_prior_pf_convergence.png', dpi=300)

# --------------------------------------------------------------------------- #
