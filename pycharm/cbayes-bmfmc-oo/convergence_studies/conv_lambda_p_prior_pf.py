# Standard stuff
import numpy as np
import matplotlib.pyplot as plt

# Model stuff
import lambda_p

# Framework stuff
from Distribution import Distribution
from Model import Model
from BMFMC import BMFMC
from CBayes import CBayesPosterior

# ------------------------------------------------- Config - General -------- #

np.random.seed(42)

# Number of lowest-fidelity samples (1e4 should be fine for 1 QoI)
n_mc_ref = int(1e4)

# ----------------------------------------------------Config - BMFMC -------- #

# Training set selection strategy (support_covering, support_covering_adaptive, sampling, sampling_adaptive)
training_set_strategy = 'support_covering'

# Regression model type (gaussian_process, heteroscedastic_gaussian_process)
regression_type = 'gaussian_process'

# ------------------------------------------------------------- Main -------- #

if __name__ == '__main__':

    n_fac_mf = 2
    n_fac_lf = 100

    n_evals_mc = np.logspace(np.log10(5), np.log10(1000), 15)
    n_evals_mc = np.round(n_evals_mc).astype(int)
    n_evals_bmfmc_hf = np.logspace(np.log10(5), np.log10(200), 10)
    n_evals_bmfmc_hf = np.round(n_evals_bmfmc_hf).astype(int)
    n_evals_bmfmc_mf = n_fac_mf * n_evals_bmfmc_hf
    n_evals_bmfmc_lf = [n_mc_ref] * 10
    n_evals_all = n_evals_mc.tolist() + list(set(n_evals_bmfmc_hf.tolist()) - set(n_evals_mc.tolist()))
    n_evals_all.sort()

    costs_hf = n_evals_bmfmc_hf
    costs_mf = 0.0 * costs_hf
    costs_lf = 0.0 * costs_mf
    total_costs_1lf = costs_hf + costs_lf
    total_costs_1lf = np.round(total_costs_1lf).astype(int)
    total_costs_2lf = costs_hf + costs_mf + costs_lf
    total_costs_2lf = np.round(total_costs_2lf).astype(int)

    # lambda_p model
    n_qoi = 1
    prior_samples = lambda_p.get_prior_samples(n_mc_ref)

    # Create the MC reference model
    print('\nCalculating MC reference ...')
    p_hf = 5
    p_mf = 3
    p_lf = 1
    ref_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_hf), rv_samples=prior_samples,
                      n_evals=n_mc_ref, n_qoi=n_qoi, rv_name='$Q$', label='MC reference')

    # Brute force Monte Carlo
    ref_prior_pf_samples = ref_model.evaluate()

    # Prior
    p_prior = Distribution(prior_samples, rv_name='$\lambda$', label='Prior', kde=False)

    # Prior push-forward
    ref_p_prior_pf = Distribution(ref_prior_pf_samples, rv_name='$Q$', label='Prior-PF')

    # -------------- 1 HF

    kls_prior_pf_1hf = []
    for idx, n_evals in enumerate(n_evals_mc):
        print('\nCalculating MC model %d / %d ...' % (idx + 1, len(n_evals_mc)))
        model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_hf), rv_samples=prior_samples[0:n_evals, :],
                      n_evals=n_evals, n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

        # Brute force Monte Carlo
        prior_pf_samples = model.evaluate()
        p_prior_pf = Distribution(prior_pf_samples, rv_name='$Q$', label='Prior-PF')

        # kl between prior push-forward and reference push-forward
        kls_prior_pf_1hf.append(ref_p_prior_pf.calculate_kl_divergence(p_prior_pf))

    # -------------- 1 HF, 1 LF

    kls_prior_pf_1hf_1lf = []
    for idx, n_evals in enumerate(n_evals_bmfmc_hf):
        print('\nCalculating BMFMC model %d / %d ...' % (idx + 1, len(n_evals_bmfmc_hf)))
        n_evals = [n_evals_bmfmc_lf[idx], n_evals]

        # Create a low-fidelity model
        lf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_lf), rv_samples=prior_samples[:n_evals[0]],
                         rv_samples_pred=prior_samples[:n_evals[0]], n_evals=n_evals[0], n_qoi=n_qoi,
                         rv_name='$q_0$', label='Low-fidelity')

        # Create a high-fidelity model
        hf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_hf), n_evals=n_evals[-1],
                         n_qoi=n_qoi,
                         rv_name='$Q$', label='High-fidelity')

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
        lf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_lf), rv_samples=prior_samples[:n_evals[0]],
                         rv_samples_pred=prior_samples[:n_evals[0]], n_evals=n_evals[0], n_qoi=n_qoi,
                         rv_name='$q_0$', label='Low-fidelity')

        # Create a mid-fidelity model
        mf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_mf), n_evals=n_evals[1],
                         n_qoi=n_qoi, rv_name='$q_1$', label='Mid-fidelity')

        # Create a high-fidelity model
        hf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_hf), n_evals=n_evals[-1],
                         n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

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
    plt.xlabel('No. HF samples')
    plt.ylabel('KL')
    plt.legend(loc='upper right')
    plt.grid(b=True)
    plt.gcf().savefig('lambda_p_prior_pf_convergence.eps', dpi=300)

# --------------------------------------------------------------------------- #
