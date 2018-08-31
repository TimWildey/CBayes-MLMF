# Standard stuff
import numpy as np
import matplotlib.pyplot as plt

# Model stuff
import elliptic_pde

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
regression_type = 'heteroscedastic_gaussian_process'

# ------------------------------------------------------------- Main -------- #

if __name__ == '__main__':

    n_evals_all = [[5], [10], [15], [20], [30], [50], [100], [200], [300], [500]]

    # elliptic_pde model
    n_qoi = 1
    obs_loc = 0.7
    obs_scale = 0.01

    # Load dataset
    lam, qvals = elliptic_pde.load_data()

    # Data split for surrogate model creation
    split = 0.05

    # Remaining data
    prior_samples = lam[round(split * lam.shape[0]) + 1:, :]
    prior_pf_samples = qvals[round(split * qvals.shape[0]) + 1:, :]

    # Choose at most the remaining number of samples
    n_samples = min(round((1 - split) * len(qvals)) - 1, n_samples)
    indices = np.random.choice(range(prior_pf_samples.shape[0]), size=n_samples, replace=False)
    prior_samples = prior_samples[indices, :]

    # Create the MC reference model
    print('\nCalculating MC reference ...')
    ref_prior_pf_samples = prior_pf_samples[indices, :]

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

    # kl
    cbayes_post = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=ref_p_prior_pf)
    cbayes_post.setup_posterior_and_pf()
    ref_kl = cbayes_post.get_prior_post_kl()

    # -------------- 1 HF

    errors_1hf = []
    kls_1hf = []
    for idx, n_evals in enumerate(n_evals_all):
        print('\nCalculating MC model %d / %d ...' % (idx + 1, len(n_evals_all)))
        prior_pf_samples_tmp = prior_pf_samples[np.random.choice(range(prior_pf_samples.shape[0]),
                                                                 size=n_evals[0], replace=False), :]
        p_prior_pf = Distribution(prior_pf_samples_tmp, rv_name='$Q$', label='Prior-PF')

        # l1 error
        l1_error = 1/n_samples * np.linalg.norm(
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

        # Construct low-fi model
        X_train = lam[0:round(split * lam.shape[0])]
        y_train = qvals[0:round(split * lam.shape[0])]
        lf_samples = elliptic_pde.construct_lowfi_model_and_get_samples(X_train=X_train, y_train=y_train,
                                                                        X_test=prior_samples)
        lf_samples = np.sin(lf_samples)

        lf_model = Model(eval_fun=lambda x: elliptic_pde.find_xy_pair(x, prior_samples, lf_samples),
                         rv_samples=prior_samples, rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                         rv_name='$q$', label='Low-fidelity')

        # High-fi model samples
        hf_samples = prior_pf_samples[indices, :]

        hf_model = Model(eval_fun=lambda x: elliptic_pde.find_xy_pair(x, prior_samples, hf_samples),
                         n_evals=n_evals[-1], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

        models = [lf_model, hf_model]

        # Setup BMFMC
        bmfmc = BMFMC(models=models,
                      training_set_strategy=training_set_strategy, regression_type=regression_type)

        # Apply BMFMC
        bmfmc.apply_bmfmc_framework()
        prior_pf_samples_tmp = bmfmc.get_samples()[-1, :, :]
        p_prior_pf = Distribution(prior_pf_samples_tmp, rv_name='$Q$', label='Prior-PF')

        # l1 error
        l1_error = 1 / n_samples * np.linalg.norm(
            p_obs_evals / ref_p_prior_pf_evals - p_obs_evals / p_prior_pf.kernel_density(ref_prior_pf_samples), 1)
        errors_1hf_1lf.append(l1_error)

        # kl
        p_obs = Distribution(obs_samples, rv_name='$Q$', label='Observed')
        cbayes_post = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf)
        cbayes_post.setup_posterior_and_pf()
        kls_1hf_1lf.append(cbayes_post.get_prior_post_kl())

    # Plots
    plt.figure()
    plt.semilogx(np.squeeze(n_evals_all), errors_1hf, '-o', label='1 HF')
    plt.semilogx(np.squeeze(n_evals_all), errors_1hf_1lf, '-o', label='1 HF, 1 LF')
    plt.semilogx(np.squeeze(n_evals_all), np.zeros((len(n_evals_all))), 'k--')
    plt.xlabel('No. high-fidelity samples')
    plt.ylabel('L1-error in the posterior')
    plt.ylim([-0.1, 1.0])
    plt.legend(loc='upper right')
    plt.gcf().savefig('convergence_studies/l1_error_conv_elliptic_pde.png', dpi=300)

    plt.figure()
    plt.semilogx(np.squeeze(n_evals_all), kls_1hf, '-o', label='1 HF')
    plt.semilogx(np.squeeze(n_evals_all), kls_1hf_1lf, '-o', label='1 HF, 1 LF')
    plt.semilogx(np.squeeze(n_evals_all), ref_kl * np.ones((len(n_evals_all))), 'k--')
    plt.xlabel('No. high-fidelity samples')
    plt.ylabel('Post-prior KL')
    # plt.ylim([-0.1, 1.0])
    plt.legend(loc='upper right')
    plt.gcf().savefig('convergence_studies/post_prior_kls_elliptic_pde.png', dpi=300)

# --------------------------------------------------------------------------- #
