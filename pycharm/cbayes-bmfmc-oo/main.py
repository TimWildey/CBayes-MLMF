# Standard stuff
import numpy as np

# Model stuff
import lambda_p
import elliptic_pde
import ode_pp

# Framework stuff
from Distribution import Distribution
from Model import Model
from BMFMC import BMFMC
from CBayes import CBayesPosterior

# ------------------------------------------------- Config - General -------- #

# Number of lowest-fidelity samples (1e4 should be fine for 1 QoI)
n_samples = int(1e4)

# Forward model (lambda_p, ellptic_pde, ode_pp)
model = 'elliptic_pde'

# Push-forward method (mc, bmfmc)
pf_method = 'bmfmc'

# ----------------------------------------------------Config - BMFMC -------- #

# Number of model evaluations in increasing fidelity (the lowest-fidelity model will always have n_samples evals)
# Only lambda_p and ode_pp support more than one (i.e. arbitrary many) low-fidelity level
# The number of models will thus be len(n_evals) + 1
# n_evals = [500, 100, 20]
# n_evals = [100, 10]
n_evals = [10]

# Training set selection strategy (support_covering, sampling, sampling_adaptive)
training_set_strategy = 'sampling_adaptive'

# Regression model type (gaussian_process)
regression_type = 'gaussian_process'

# ---------------------------------------------------------- Todos ---------- #

# Adaptive training
# todo: check why alternative KL calculation fails
# todo: create unique training sets
# todo: implement support_covering_adaptive
# todo: different options where to choose samples
# todo: uniform / LHS / sparse grids

# Distributions
# todo: store KDE evaluations for distributions in a vector for reuse
# todo: implement distribution transformations to operate in unconstrained probability space only

# CBayes
# todo: why is cbayes so robust wrt. the push-forward of the prior?
# todo: add support for multiple QoIs (some parts are already there)

# BMFMC
# todo: check how to deal with the case where one has fixed evaluation points
# todo: implement other regression models

# ------------------------------------------------- Models & Methods -------- #

def get_prior_prior_pf_samples(n_samples):

    prior_samples = prior_pf_samples = []

    if model == 'lambda_p':

        n_qoi = 1
        obs_loc = 0.25
        obs_scale = 0.1
        prior_samples = lambda_p.get_prior_samples(n_samples)

        if pf_method == 'mc':

            # Create the high-fidelity model
            hf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, 5), rv_samples=prior_samples,
                             n_evals=n_samples, n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            # Brute force Monte Carlo
            prior_pf_samples = hf_model.evaluate()

        elif pf_method == 'bmfmc':

            # Create a low-fidelity model
            lf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, 1), rv_samples=prior_samples,
                             rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                             rv_name='$q_0$', label='Low-fidelity')

            # Create a high-fidelity model
            hf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, len(n_evals) * 2 + 1), n_evals=n_evals[-1],
                             n_qoi=n_qoi,
                             rv_name='$Q$', label='High-fidelity')

            if len(n_evals) == 2:
                # Create a mid fidelity model
                mf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, 3), n_evals=n_evals[0], n_qoi=n_qoi,
                                 rv_name='$q_m$', label='Mid-fidelity')
                models = [lf_model, mf_model, hf_model]

            elif len(n_evals) > 2:
                models = [lf_model]
                for i in range(len(n_evals) - 1):
                    models.append(
                        Model(eval_fun=lambda x, i=i: lambda_p.lambda_p(x, (i + 1) * 2 + 1), n_evals=n_evals[i],
                              n_qoi=n_qoi, rv_name='$q_%d$' % int(i + 1), label='Mid-%d-fidelity' % int(i + 1)))
                models.append(hf_model)

            elif len(n_evals) == 1:
                models = [lf_model, hf_model]

            else:
                print('Unsupported number of models for lambda_p.')
                exit()

            # Setup BMFMC
            bmfmc = BMFMC(models=models,
                          training_set_strategy=training_set_strategy, regression_type=regression_type)

            # Apply BMFMC
            bmfmc.apply_bmfmc_framework()

            # Calculate Monte Carlo reference
            bmfmc.calculate_mc_reference()

            # Diagnostics
            bmfmc.print_stats(mc=True)
            bmfmc.plot_results(mc=True)
            bmfmc.plot_regression_models()
            bmfmc.plot_joint_densities()

            # Get prior push-forward samples
            prior_pf_samples = bmfmc.get_high_fidelity_samples()
            prior_pf_lf_samples = bmfmc.get_low_fidelity_samples()

        else:
            print('Unknown push-forward method: %r' % pf_method)
            exit()

    elif model == 'elliptic_pde':

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
                             rv_name='$q$', label='Low-fidelity')

            # High-fi model samples
            hf_samples = prior_pf_samples[indices, :]

            hf_model = Model(eval_fun=lambda x: elliptic_pde.find_xy_pair(x, prior_samples, hf_samples),
                             n_evals=n_evals[-1], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            # Setup BMFMC
            bmfmc = BMFMC(models=[lf_model, hf_model],
                          training_set_strategy=training_set_strategy, regression_type=regression_type)

            # Apply BMFMC
            bmfmc.apply_bmfmc_framework()

            # Calculate Monte Carlo reference
            bmfmc.calculate_mc_reference()

            # Diagnostics
            bmfmc.print_stats(mc=True)
            bmfmc.plot_results(mc=True)
            bmfmc.plot_regression_models()
            bmfmc.plot_joint_densities()

            # Get prior push-forward samples
            prior_pf_samples = bmfmc.get_high_fidelity_samples()
            prior_pf_lf_samples = bmfmc.get_low_fidelity_samples()

        else:
            print('Unknown push-forward method: %r' % pf_method)
            exit()

    elif model == 'ode_pp':

        n_qoi = 1
        obs_loc = 2.5
        obs_scale = 0.1
        prior_samples = ode_pp.get_prior_samples(n_samples)

        # Model settings
        u0 = np.array([5, 1])
        finalt = 1.0
        dt_hf = 0.5

        hf_settings = ode_pp.Settings(finalt=finalt, dt=dt_hf, u0=u0)

        if pf_method == 'mc':

            # Create the high-fidelity model
            hf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, hf_settings)[0, -1], rv_samples=prior_samples,
                             n_evals=n_samples, n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            # Brute force Monte Carlo
            prior_pf_samples = hf_model.evaluate()

        elif pf_method == 'bmfmc':

            # Create a low-fidelity model
            dt_lf = 1.0
            lf_settings = ode_pp.Settings(finalt=finalt, dt=dt_lf, u0=u0)
            lf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, lf_settings)[0, -1], rv_samples=prior_samples,
                             rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                             rv_name='$q_0$', label='Low-fidelity')

            # Create a high-fidelity model
            hf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, hf_settings)[0, -1], n_evals=n_evals[-1], n_qoi=n_qoi,
                             rv_name='$Q$', label='High-fidelity')

            if len(n_evals) == 2:
                # Create a mid fidelity model
                dt_mf = 0.7
                mf_settings = ode_pp.Settings(finalt=finalt, dt=dt_mf, u0=u0)
                mf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, mf_settings)[0, -1], n_evals=n_evals[0],
                                 n_qoi=n_qoi, rv_name='$q_m$', label='Mid-fidelity')
                models = [lf_model, mf_model, hf_model]

            elif len(n_evals) > 2:
                models = [lf_model]
                dts = np.linspace(dt_lf, dt_hf, len(n_evals) + 2)
                # dts = [dt_lf, 0.9, 0.6, dt_hf]
                for i in range(len(n_evals) - 1):
                    settings = ode_pp.Settings(finalt=finalt, dt=dts[i + 1], u0=u0)
                    models.append(Model(eval_fun=lambda x, settings=settings: ode_pp.ode_pp(x, settings)[0, -1],
                                        n_evals=n_evals[i],
                                        n_qoi=n_qoi, rv_name='$q_%d$' % int(i + 1),
                                        label='Mid-%d-fidelity' % int(i + 1)))
                models.append(hf_model)

            elif len(n_evals) == 1:
                models = [lf_model, hf_model]

            else:
                print('Unsupported number of models for ode_pp.')
                exit()

            # Setup BMFMC
            bmfmc = BMFMC(models=models,
                          training_set_strategy=training_set_strategy, regression_type=regression_type)

            # Apply BMFMC
            bmfmc.apply_bmfmc_framework()

            # Calculate Monte Carlo reference
            bmfmc.calculate_mc_reference()

            # Diagnostics
            bmfmc.print_stats(mc=True)
            bmfmc.plot_results(mc=True)
            bmfmc.plot_regression_models()
            bmfmc.plot_joint_densities()

            # Get prior push-forward samples
            prior_pf_samples = bmfmc.get_high_fidelity_samples()
            prior_pf_lf_samples = bmfmc.get_low_fidelity_samples()

        else:
            print('Unknown push-forward method: %r' % pf_method)
            exit()

    else:
        print('Unknown model: %r' % model)
        exit()

    return prior_samples, prior_pf_samples, obs_loc, obs_scale, prior_pf_lf_samples


# --------------------------------------------------------------------------- #


# ------------------------------------------------------------- Main -------- #

if __name__ == '__main__':

    # Get samples from the prior, its push-forward and the observed density
    print('')
    print('Calculating the Prior push-forward ...')
    prior_samples, prior_pf_samples, obs_loc, obs_scale, prior_pf_lf_samples = get_prior_prior_pf_samples(n_samples)

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
    cbayes_post.setup_posterior_and_pf()

    # Output
    cbayes_post.print_stats()

    # Plot
    cbayes_post.plot_results(1)

    # Low-fidelity comparison
    if pf_method == 'bmfmc':
        p_prior_pf_lf = Distribution(prior_pf_lf_samples, rv_name='$Q$', label='Prior-PF-LF')
        obs_samples = np.random.randn(n_samples) * obs_scale + obs_loc
        obs_samples = np.reshape(obs_samples, (n_samples, 1))
        p_obs = Distribution(obs_samples, rv_name='$Q$', label='Observed')
        cbayes_post_lf = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf_lf)
        cbayes_post_lf.setup_posterior_and_pf()
        print('Evaluating a low-fidelity posterior ...')
        cbayes_post_lf.print_stats()
        cbayes_post_lf.plot_results(2)

# --------------------------------------------------------------------------- #
