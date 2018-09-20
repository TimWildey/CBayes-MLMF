# Standard stuff
import numpy as np
import matplotlib.pyplot as plt

# Model stuff
import lambda_p

# Framework stuff
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

    n_evals_mc = np.logspace(np.log10(10), np.log10(2000), 15)
    n_evals_mc = np.round(n_evals_mc).astype(int)
    n_evals_bmfmc = np.logspace(np.log10(10), np.log10(200), 10)
    n_evals_bmfmc = np.round(n_evals_bmfmc).astype(int)
    n_evals_all = n_evals_mc.tolist() + list(set(n_evals_bmfmc.tolist()) - set(n_evals_mc.tolist()))
    n_evals_all.sort()

    # lambda_p model
    n_qoi = 1
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
    mc_mean_ref = np.mean(ref_prior_pf_samples, axis=0)

    # -------------- MC

    mc_error = []
    mc_mean = []
    for idx, n_evals in enumerate(n_evals_mc):
        print('\nCalculating MC model %d / %d ...' % (idx + 1, len(n_evals_mc)))
        model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_hf), rv_samples=prior_samples[0:n_evals, :],
                      n_evals=n_evals, n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')
        prior_pf_samples = model.evaluate()

        # MC error
        mc_error.append(np.sqrt(np.var(prior_pf_samples) / n_evals))
        mc_mean.append(np.mean(prior_pf_samples))

    # -------------- BMFMC

    bmfmc_error = []
    bmfmc_mean = []
    for idx, n_evals in enumerate(n_evals_bmfmc):
        print('\nCalculating BMFMC model %d / %d ...' % (idx + 1, len(n_evals_bmfmc)))

        # Create a low-fidelity model
        lf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_lf), rv_samples=prior_samples,
                         rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                         rv_name='$q_0$', label='Low-fidelity')

        # Create a high-fidelity model
        hf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_hf), n_evals=n_evals,
                         n_qoi=n_qoi,
                         rv_name='$Q$', label='High-fidelity')

        models = [lf_model, hf_model]

        # Setup BMFMC
        bmfmc = BMFMC(models=models,
                      training_set_strategy=training_set_strategy, regression_type=regression_type)

        # Apply BMFMC
        bmfmc.apply_bmfmc_framework()
        prior_pf_samples = bmfmc.get_samples()[-1, :, :]

        # BMFMC error
        bmfmc_error.append(np.sqrt(bmfmc.calculate_bmfmc_mean_estimator_variance()))
        bmfmc_mean.append(np.mean(prior_pf_samples))

    # -------------- BMFMC 2 lf

    bmfmc2_error = []
    bmfmc2_mean = []
    for idx, n_evals in enumerate(n_evals_bmfmc):
        n_evals = [2*n_evals, n_evals]
        print('\nCalculating BMFMC model %d / %d ...' % (idx + 1, len(n_evals_bmfmc)))

        # Create a low-fidelity model
        lf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_lf), rv_samples=prior_samples,
                         rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                         rv_name='$q_0$', label='Low-fidelity')

        # Create a mid-fidelity model
        mf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_mf), n_evals=n_evals[0], n_qoi=n_qoi,
                         rv_name='$q_1$', label='Mid-fidelity')

        # Create a high-fidelity model
        hf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, p_hf), n_evals=n_evals[-1], n_qoi=n_qoi,
                         rv_name='$Q$', label='High-fidelity')

        models = [lf_model, mf_model, hf_model]

        # Setup BMFMC
        bmfmc = BMFMC(models=models,
                      training_set_strategy=training_set_strategy, regression_type=regression_type)

        # Apply BMFMC
        bmfmc.apply_bmfmc_framework()
        prior_pf_samples = bmfmc.get_samples()[-1, :, :]

        # BMFMC error
        bmfmc2_error.append(np.sqrt(bmfmc.calculate_bmfmc_mean_estimator_variance()))
        bmfmc2_mean.append(np.mean(prior_pf_samples))

    mc_error = np.array(mc_error)
    mc_mean = np.array(mc_mean)
    bmfmc_error = np.array(bmfmc_error)
    bmfmc_mean = np.array(bmfmc_mean)
    bmfmc2_error = np.array(bmfmc2_error)
    bmfmc2_mean = np.array(bmfmc2_mean)

    # Plots
    plt.figure()
    plt.semilogx(np.squeeze(n_evals_mc), mc_error, '-o', label='MC')
    plt.semilogx(np.squeeze(n_evals_bmfmc), bmfmc_error, '-o', label='BMFMC')
    plt.semilogx(np.squeeze(n_evals_bmfmc), bmfmc2_error, '-o', label='BMFMC2')
    plt.semilogx(np.squeeze(n_evals_all), np.zeros((len(n_evals_all))), 'k--')
    plt.xlabel('No. high-fidelity samples')
    plt.ylabel('Estimator error')
    plt.legend(loc='upper right')
    plt.gcf().savefig('lambda_p_estimator_errors.png', dpi=300)

    plt.figure()
    plt.semilogx(np.squeeze(n_evals_mc), mc_mean + 3 * mc_error, 'C0--')
    plt.semilogx(np.squeeze(n_evals_mc), mc_mean - 3 * mc_error, 'C0--')
    plt.semilogx(np.squeeze(n_evals_mc), mc_mean, 'C0-o', label='MC')
    plt.semilogx(np.squeeze(n_evals_bmfmc), bmfmc_mean + 3 * bmfmc_error, 'C1--')
    plt.semilogx(np.squeeze(n_evals_bmfmc), bmfmc_mean - 3 * bmfmc_error, 'C1--')
    plt.semilogx(np.squeeze(n_evals_bmfmc), bmfmc_mean, 'C1-o', label='BMFMC')
    plt.semilogx(np.squeeze(n_evals_bmfmc), bmfmc2_mean + 3 * bmfmc2_error, 'C2--')
    plt.semilogx(np.squeeze(n_evals_bmfmc), bmfmc2_mean - 3 * bmfmc2_error, 'C2--')
    plt.semilogx(np.squeeze(n_evals_bmfmc), bmfmc2_mean, 'C2-o', label='BMFMC2')
    plt.semilogx(np.squeeze(n_evals_all), mc_mean_ref * np.ones((len(n_evals_all))), 'k--', label='reference')
    plt.xlabel('No. high-fidelity samples')
    plt.ylabel('Mean + error bounds')
    plt.legend(loc='upper right')
    plt.gcf().savefig('lambda_p_estimator_mean.png', dpi=300)

# --------------------------------------------------------------------------- #
