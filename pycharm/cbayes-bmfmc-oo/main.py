# Standard stuff
import numpy as np

# Framework stuff
from Distribution import Distribution
from Model import Model
from CBayes import CBayesPosterior
from BMFMC import BMFMC

# Model stuff
import lambda_p
import elliptic_pde
import ode_pp

# ----------------------------------------------------------- Config -------- #
#
# General
#
n_samples = int(1e4)                            # 1e4 should suffice
model = 'ode_pp'                                # lambda_p, elliptic_pde, ode_pp
pf_method = 'bmfmc'                             # mc, bmfmc
obs_loc = 2.5                                   # lambda_p: 0.25, elliptic_pde: 0.7, ode_pp: 2.5
obs_scale = 0.1                                 # lambda_p: 0.1, elliptic_pde: 0.01, ode_pp: 0.1
#
# BMFMC specific
#
n_evals = 50                                    # [20, 100] (the lowest-fidelity model will always have n_samples evals)
training_set_strategy = 'support_covering'      # support_covering, sampling
regression_type = 'gaussian_process'            # gaussian_process
#
# --------------------------------------------------------------------------- #

# todo: add support for a hierarchy of low-fidelity models (some parts are already there)
# todo: add support for multiple QoIs (some parts are already there)

# ------------------------------------------------- Models & Methods -------- #
#
def get_prior_prior_pf_samples(n_samples):
    prior_samples = prior_pf_samples = []

    if model == 'lambda_p':
        n_qoi = 1
        prior_samples = lambda_p.get_prior_samples(n_samples)

        if pf_method == 'mc':

            # Create the high-fidelity model
            hf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, 5), rv_samples=prior_samples,
                             n_evals=n_samples, n_qoi=n_qoi, rv_name='Q', label='High-fidelity')

            # Brute force Monte Carlo
            prior_pf_samples = hf_model.evaluate()

        elif pf_method == 'bmfmc':

            # Create a low-fidelity model
            lf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, 3), rv_samples=prior_samples,
                             rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                             rv_name='q', label='Low-fidelity')

            # Create a high-fidelity model
            hf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, 5), n_evals=n_evals, n_qoi=n_qoi,
                             rv_name='Q', label='High-fidelity')

            # Setup BMFMC
            bmfmc = BMFMC(models=[lf_model, hf_model],
                          training_set_strategy=training_set_strategy, regression_type=regression_type)

            # Apply BMFMC
            bmfmc.apply_bmfmc_framework()

            # Calculate Monte Carlo reference
            bmfmc.calculate_mc_reference()

            # Diagnostics
            bmfmc.print_stats_with_mc()
            bmfmc.plot_results_with_mc()
            bmfmc.plot_regression_models()
            bmfmc.plot_joint_densities()

            # Get prior push-forward samples
            prior_pf_samples = bmfmc.get_high_fidelity_samples()

        else:
            print('Unknown push-forward method: %r' % pf_method)
            exit()

    elif model == 'elliptic_pde':

        n_qoi = 1

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

        if pf_method == 'mc':

            # Monte Carlo reference
            prior_pf_samples = prior_pf_samples[np.random.choice(range(prior_pf_samples.shape[0]),
                                                                 size=n_samples, replace=False), :]

        elif pf_method == 'bmfmc':

            # Construct low-fi model
            X_train = lam[0:round(split * lam.shape[0])]
            y_train = qvals[0:round(split * lam.shape[0])]
            lf_samples = elliptic_pde.construct_lowfi_model_and_get_samples(X_train=X_train, y_train=y_train,
                                                                            X_test=prior_samples)
            lf_samples = np.sin(lf_samples)

            lf_model = Model(eval_fun=lambda x: elliptic_pde.find_xy_pair(x, prior_samples, lf_samples),
                             rv_samples=prior_samples, rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                             rv_name='q', label='Low-fidelity')

            # High-fi model samples
            hf_samples = prior_pf_samples[indices, :]

            hf_model = Model(eval_fun=lambda x: elliptic_pde.find_xy_pair(x, prior_samples, hf_samples),
                             n_evals=n_evals, n_qoi=n_qoi, rv_name='Q', label='High-fidelity')

            # Setup BMFMC
            bmfmc = BMFMC(models=[lf_model, hf_model],
                          training_set_strategy=training_set_strategy, regression_type=regression_type)

            # Apply BMFMC
            bmfmc.apply_bmfmc_framework()

            # Calculate Monte Carlo reference
            bmfmc.calculate_mc_reference()

            # Diagnostics
            bmfmc.print_stats_with_mc()
            bmfmc.plot_results_with_mc()
            bmfmc.plot_regression_models()
            bmfmc.plot_joint_densities()

            # Get prior push-forward samples
            prior_pf_samples = bmfmc.get_high_fidelity_samples()

        else:
            print('Unknown push-forward method: %r' % pf_method)
            exit()

    elif model == 'ode_pp':

        n_qoi = 1
        prior_samples = ode_pp.get_prior_samples(n_samples)

        # Model settings
        u0 = np.array([5, 1])
        finalt = 1.0
        dt_hf = 0.01
        dt_lf = 0.9

        hf_settings = ode_pp.Settings(finalt=finalt, dt=dt_hf, u0=u0)

        if pf_method == 'mc':

            # Create the high-fidelity model
            hf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, hf_settings)[0, -1], rv_samples=prior_samples,
                             n_evals=n_samples, n_qoi=n_qoi, rv_name='Q', label='High-fidelity')

            # Brute force Monte Carlo
            prior_pf_samples = hf_model.evaluate()

        elif pf_method == 'bmfmc':

            # Create a low-fidelity model
            lf_settings = ode_pp.Settings(finalt=finalt, dt=dt_lf, u0=u0)
            lf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, lf_settings)[0, -1], rv_samples=prior_samples,
                             rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                             rv_name='q', label='Low-fidelity')

            # Create a high-fidelity model
            hf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, hf_settings)[0, -1], n_evals=n_evals, n_qoi=n_qoi,
                             rv_name='Q', label='High-fidelity')

            # Setup BMFMC
            bmfmc = BMFMC(models=[lf_model, hf_model],
                          training_set_strategy=training_set_strategy, regression_type=regression_type)

            # Apply BMFMC
            bmfmc.apply_bmfmc_framework()

            # Calculate Monte Carlo reference
            bmfmc.calculate_mc_reference()

            # Diagnostics
            bmfmc.print_stats_with_mc()
            bmfmc.plot_results_with_mc()
            bmfmc.plot_regression_models()
            bmfmc.plot_joint_densities()

            # Get prior push-forward samples
            prior_pf_samples = bmfmc.get_high_fidelity_samples()

        else:
            print('Unknown push-forward method: %r' % pf_method)
            exit()

    else:
        print('Unknown model: %r' % model)
        exit()

    return prior_samples, prior_pf_samples
#
# --------------------------------------------------------------------------- #


# ------------------------------------------------------------- Main -------- #
#
if __name__ == '__main__':

    # Get samples from the prior, its push-forward and the observed density
    print('\nCalculating the Prior push-forward ...')
    prior_samples, prior_pf_samples = get_prior_prior_pf_samples(n_samples)

    # Prior
    p_prior = Distribution(prior_samples, rv_name='$\lambda$', label='Prior', kde=False)

    # Prior push-forward
    p_prior_pf = Distribution(prior_pf_samples, rv_name='$Q$', label='Prior-PF')

    # Observed density
    obs_samples = np.random.randn(n_samples) * obs_scale + obs_loc
    obs_samples = np.reshape(obs_samples, (n_samples, 1))
    p_obs = Distribution(obs_samples, rv_name='$Q$', label='Observed')

    # Posterior
    print('Evaluating the posterior ...')
    cbayes_post = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf)
    p_post, p_post_pf = cbayes_post.setup_posterior_and_pf()

    # Output
    cbayes_post.print_stats()

    # Plot
    cbayes_post.plot_results(1)
#
# --------------------------------------------------------------------------- #
