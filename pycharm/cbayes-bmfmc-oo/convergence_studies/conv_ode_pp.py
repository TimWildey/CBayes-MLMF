# Standard stuff
import numpy as np
import matplotlib.pyplot as plt

# Model stuff
import ode_pp

# Framework stuff
from Distribution import Distribution
from Model import Model
from BMFMC import BMFMC
from CBayes import CBayesPosterior

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

    # ode_pp model
    n_qoi = 1
    obs_loc = 2.5
    obs_scale = 0.1
    prior_samples = ode_pp.get_prior_samples(n_samples)

    # Model settings
    u0 = np.array([5, 1])
    finalt = 1.0
    dt_hf = 0.1

    ref_settings = ode_pp.Settings(finalt=finalt, dt=dt_hf, u0=u0)

    # Create the MC reference model
    print('\nCalculating MC reference ...')
    ref_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, ref_settings)[0, -1], rv_samples=prior_samples,
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

    # pre evaluations
    ref_prior_pf_samples = np.squeeze(ref_prior_pf_samples)
    p_obs_evals = p_obs.kernel_density(ref_prior_pf_samples)
    ref_p_prior_pf_evals = ref_p_prior_pf.kernel_density(ref_prior_pf_samples)

    # kl
    cbayes_post = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=ref_p_prior_pf)
    cbayes_post.setup_posterior_and_pf()
    ref_kl = cbayes_post.get_prior_post_kl()

    # -------------- 1 HF

    errors_1hf = []
    kls_1hf = []
    for idx, n_evals in enumerate(n_evals_all):
        print('\nCalculating MC model %d / %d ...' % (idx + 1, len(n_evals_all)))
        model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, ref_settings)[0, -1],
                      rv_samples=prior_samples[0:n_evals[0], :],
                      n_evals=n_evals[0], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

        # Brute force Monte Carlo
        prior_pf_samples = model.evaluate()
        p_prior_pf = Distribution(prior_pf_samples, rv_name='$Q$', label='Prior-PF')

        # l1 error
        l1_error = 1 / n_samples * np.linalg.norm(
            p_obs_evals / ref_p_prior_pf_evals - p_obs_evals / p_prior_pf.kernel_density(ref_prior_pf_samples), 1)
        errors_1hf.append(l1_error)

        # kl
        p_obs = Distribution(obs_samples, rv_name='$Q$', label='Observed')
        cbayes_post = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf)
        cbayes_post.setup_posterior_and_pf()
        kls_1hf.append(cbayes_post.get_prior_post_kl())

    # -------------- 1 HF, 1 LF

    errors_1hf_1lf = []
    kls_1hf_1lf = []
    for idx, n_evals in enumerate(n_evals_all):
        print('\nCalculating BMFMC model %d / %d ...' % (idx + 1, len(n_evals_all)))

        # Create a low-fidelity model
        dt_lf = 1.0
        lf_settings = ode_pp.Settings(finalt=finalt, dt=dt_lf, u0=u0)
        lf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, lf_settings)[0, -1], rv_samples=prior_samples,
                         rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                         rv_name='$q_0$', label='Low-fidelity')

        # Create a high-fidelity model
        hf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, ref_settings)[0, -1], n_evals=n_evals[-1], n_qoi=n_qoi,
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

        # kl
        p_obs = Distribution(obs_samples, rv_name='$Q$', label='Observed')
        cbayes_post = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf)
        cbayes_post.setup_posterior_and_pf()
        kls_1hf_1lf.append(cbayes_post.get_prior_post_kl())

    # -------------- 1 HF, 2 LF

    errors_1hf_2lf = []
    kls_1hf_2lf = []
    n_low = n_evals_all[-1][0]
    for idx, n_evals in enumerate(n_evals_all):
        n_evals = [n_low, n_evals[0]]
        print('\nCalculating BMFMC multi-model %d / %d ...' % (idx + 1, len(n_evals_all)))

        # Create a low-fidelity model
        dt_lf = 1.0
        lf_settings = ode_pp.Settings(finalt=finalt, dt=dt_lf, u0=u0)
        lf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, lf_settings)[0, -1], rv_samples=prior_samples,
                         rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                         rv_name='$q_0$', label='Low-fidelity')

        # Create a mid fidelity model
        dt_mf = 0.7
        mf_settings = ode_pp.Settings(finalt=finalt, dt=dt_mf, u0=u0)
        mf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, mf_settings)[0, -1], n_evals=n_evals[0],
                         n_qoi=n_qoi, rv_name='$q_m$', label='Mid-fidelity')

        # Create a high-fidelity model
        hf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, ref_settings)[0, -1], n_evals=n_evals[-1], n_qoi=n_qoi,
                         rv_name='$Q$', label='High-fidelity')

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

        # kl
        p_obs = Distribution(obs_samples, rv_name='$Q$', label='Observed')
        cbayes_post = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf)
        cbayes_post.setup_posterior_and_pf()
        kls_1hf_2lf.append(cbayes_post.get_prior_post_kl())

    # Plots
    plt.figure()
    plt.semilogx(np.squeeze(n_evals_all), errors_1hf, '-o', label='1 HF')
    plt.semilogx(np.squeeze(n_evals_all), errors_1hf_1lf, '-o', label='1 HF, 1 LF')
    plt.semilogx(np.squeeze(n_evals_all), errors_1hf_2lf, '-o', label='1 HF, 2 LF')
    plt.semilogx(np.squeeze(n_evals_all), np.zeros((len(n_evals_all))), 'k--')
    plt.xlabel('No. high-fidelity samples')
    plt.ylabel('L1-error in the posterior')
    plt.ylim([-0.1, 1.0])
    plt.legend(loc='upper right')
    plt.gcf().savefig('convergence_studies/l1_error_conv_ode_pp.png', dpi=300)

    plt.figure()
    plt.semilogx(np.squeeze(n_evals_all), kls_1hf, '-o', label='1 HF')
    plt.semilogx(np.squeeze(n_evals_all), kls_1hf_1lf, '-o', label='1 HF, 1 LF')
    plt.semilogx(np.squeeze(n_evals_all), kls_1hf_2lf, '-o', label='1 HF, 2 LF')
    plt.semilogx(np.squeeze(n_evals_all), ref_kl * np.ones((len(n_evals_all))), 'k--')
    plt.xlabel('No. high-fidelity samples')
    plt.ylabel('Post-prior KL')
    # plt.ylim([-0.1, 1.0])
    plt.legend(loc='upper right')
    plt.gcf().savefig('convergence_studies/post_prior_kls_ode_pp.png', dpi=300)

# --------------------------------------------------------------------------- #
