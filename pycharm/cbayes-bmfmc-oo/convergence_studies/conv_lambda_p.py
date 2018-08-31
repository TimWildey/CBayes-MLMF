# Standard stuff
import numpy as np
import matplotlib.pyplot as plt

# Model stuff
import lambda_p

# Framework stuff
from Distribution import Distribution
from Model import Model
from BMFMC import BMFMC

# ------------------------------------------------- Config - General -------- #

# Number of lowest-fidelity samples (1e4 should be fine for 1 QoI)
n_samples = int(1e4)

# ----------------------------------------------------Config - BMFMC -------- #

# Training set selection strategy (support_covering, support_covering_adaptive, sampling, sampling_adaptive)
training_set_strategy = 'support_covering'

# Regression model type (gaussian_process, heteroscedastic_gaussian_process)
regression_type = 'gaussian_process'

# ------------------------------------------------------------- Main -------- #

if __name__ == '__main__':

    n_evals_all = [[5], [10], [15], [20], [30], [50], [100], [200], [300], [500]]

    # lambda_p model
    n_qoi = 1
    obs_loc = 0.25
    obs_scale = 0.1
    prior_samples = lambda_p.get_prior_samples(n_samples)

    # Create the MC reference model
    print('\nCalculating MC reference ...')
    p_hf = 5
    p_mf = 3
    p_lf = 1
    ref_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_hf), rv_samples=prior_samples,
                      n_evals=n_samples, n_qoi=n_qoi, rv_name='$Q$', label='MC reference')

    # Brute force Monte Carlo
    ref_prior_pf_samples = ref_model.evaluate()

    # Prior
    p_prior = Distribution(prior_samples, rv_name='$\lambda$', label='Prior', kde=False)

    # Prior push-forward
    ref_p_prior_pf = Distribution(ref_prior_pf_samples, rv_name='$Q$', label='Prior-PF')

    # Observed density
    obs_samples = np.random.randn(n_samples) * obs_scale + obs_loc
    obs_samples = np.reshape(obs_samples, (n_samples, 1))
    p_obs = Distribution(obs_samples, rv_name='$Q$', label='Observed')

    # Pre evaluations
    ref_prior_pf_samples = np.squeeze(ref_prior_pf_samples)
    p_obs_evals = p_obs.kernel_density(ref_prior_pf_samples)
    ref_p_prior_pf_evals = ref_p_prior_pf.kernel_density(ref_prior_pf_samples)

    # -------------- 1 HF

    errors_1hf = []
    for idx, n_evals in enumerate(n_evals_all):
        print('\nCalculating MC model %d / %d ...' % (idx + 1, len(n_evals_all)))
        model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_hf), rv_samples=prior_samples[0:n_evals[0], :],
                      n_evals=n_evals[0], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

        # Brute force Monte Carlo
        prior_pf_samples = model.evaluate()
        p_prior_pf = Distribution(prior_pf_samples, rv_name='$Q$', label='Prior-PF')

        # l1 error
        l1_error = 1 / n_samples * np.linalg.norm(
            p_obs_evals / ref_p_prior_pf_evals - p_obs_evals / p_prior_pf.kernel_density(ref_prior_pf_samples), 1)
        errors_1hf.append(l1_error)

    # -------------- 1 HF, 1 LF

    errors_1hf_1lf = []
    for idx, n_evals in enumerate(n_evals_all):
        print('\nCalculating BMFMC model %d / %d ...' % (idx + 1, len(n_evals_all)))

        # Create a low-fidelity model
        lf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_lf), rv_samples=prior_samples,
                         rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
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

        # l1 error
        l1_error = 1 / n_samples * np.linalg.norm(
            p_obs_evals / ref_p_prior_pf_evals - p_obs_evals / p_prior_pf.kernel_density(ref_prior_pf_samples), 1)
        errors_1hf_1lf.append(l1_error)

    # -------------- 1 HF, 2 LF

    n_low = n_evals_all[-1][0]
    errors_1hf_2lf = []
    for idx, n_evals in enumerate(n_evals_all):
        n_evals = [n_low, n_evals[0]]
        print('\nCalculating BMFMC multi-model %d / %d ...' % (idx + 1, len(n_evals_all)))

        # Create a low-fidelity model
        lf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_lf), rv_samples=prior_samples,
                         rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                         rv_name='$q_0$', label='Low-fidelity')

        # Create a mid-fidelity model
        mf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_mf), n_evals=n_evals[0],
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

        # l1 error
        l1_error = 1 / n_samples * np.linalg.norm(
            p_obs_evals / ref_p_prior_pf_evals - p_obs_evals / p_prior_pf.kernel_density(ref_prior_pf_samples), 1)
        errors_1hf_2lf.append(l1_error)

    # Plot
    plt.figure()
    plt.semilogx(np.squeeze(n_evals_all), errors_1hf, '-o', label='1 HF')
    plt.semilogx(np.squeeze(n_evals_all), errors_1hf_1lf, '-o', label='1 HF, 1 LF')
    plt.semilogx(np.squeeze(n_evals_all), errors_1hf_2lf, '-o', label='1 HF, 2 LF')
    plt.semilogx(np.squeeze(n_evals_all), np.zeros((len(n_evals_all))), 'k--')
    plt.xlabel('No. high-fidelity samples')
    plt.ylabel('L1-error in the posterior')
    plt.ylim([-0.1, 1.0])
    plt.legend(loc='upper right')
    plt.gcf().savefig('convergence_studies/l1_error_conv_lambda_p.png', dpi=300)

# --------------------------------------------------------------------------- #
