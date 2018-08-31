# Standard stuff
import numpy as np
import os
import time
import matplotlib.pyplot as plt

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
model = 'ode_pp'

# Push-forward method (mc, bmfmc)
pf_method = 'bmfmc'


# ----------------------------------------------------Config - BMFMC -------- #


# Number of model evaluations in increasing fidelity (the lowest-fidelity model will always have n_samples evals)
# Only lambda_p and ode_pp support more than one (i.e. arbitrary many) low-fidelity level
# The number of models will thus be len(n_evals) + 1
# n_evals = [200, 100, 50, 20, 10, 5]
n_evals = [50, 20, 5]

# Training set selection strategy (support_covering, support_covering_adaptive, sampling, sampling_adaptive)
training_set_strategy = 'support_covering_adaptive'

# Regression model type (gaussian_process, heteroscedastic_gaussian_process)
regression_type = 'heteroscedastic_gaussian_process'


# ---------------------------------------------------------- Todos ---------- #


# Validation
# todo: (!!!) create convergence plots of the l1 error of the posterior over the number of high-fidelity evaluations

# Framework
# todo: (!!!) add support for multiple QoIs (some parts are already there)
# todo: (!!) check how to deal with the case where one has fixed evaluation points --> training_set_strategy: fixed

# Distributions
# todo: (!) implement transformations of random variables to operate in unconstrained probability space only

# Regression
# todo: (!) implement other regression models

# Adaptive training
# todo: (!!) adaptive sampling using the GP predictive uncertainty --> training_set_strategy: sampling_gp_adaptive
# todo: (!) different options how to choose samples: uniform / LHS / sparse grids


# ------------------------------------------------- Models & Methods -------- #


def get_prior_prior_pf_samples(n_samples):
    prior_samples = prior_pf_samples = obs_loc = obs_scale = prior_pf_mc_samples = []

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
            prior_pf_samples = np.reshape(prior_pf_samples,
                                          (1, n_samples, np.shape(prior_pf_samples)[1]))
            prior_pf_mc_samples = prior_pf_samples

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
            prior_pf_samples = prior_pf_samples[indices, :]
            prior_pf_samples = np.reshape(prior_pf_samples,
                                          (1, n_samples, np.shape(prior_pf_samples)[1]))
            prior_pf_mc_samples = prior_pf_samples

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

            models = [lf_model, hf_model]

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
            prior_pf_samples = np.reshape(prior_pf_samples,
                                          (1, n_samples, np.shape(prior_pf_samples)[1]))
            prior_pf_mc_samples = prior_pf_samples

        elif pf_method == 'bmfmc':

            # Create a low-fidelity model
            dt_lf = 1.0
            lf_settings = ode_pp.Settings(finalt=finalt, dt=dt_lf, u0=u0)
            lf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, lf_settings)[0, -1] + 1.0, rv_samples=prior_samples,
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

        else:
            print('Unknown push-forward method: %r' % pf_method)
            exit()

    else:
        print('Unknown model: %r' % model)
        exit()

    if pf_method == 'bmfmc':
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
        # bmfmc.plot_joint_densities()

        # Get prior push-forward samples
        prior_pf_samples = bmfmc.get_samples()
        prior_pf_mc_samples = bmfmc.get_mc_samples()

    return prior_samples, prior_pf_samples, obs_loc, obs_scale, prior_pf_mc_samples


# ------------------------------------------------------------- Main -------- #


if __name__ == '__main__':

    # Timing
    start = time.time()

    # Clean pngout folder
    if os.path.isdir('pngout'):
        os.system('rm -rf pngout/')
        os.mkdir('pngout')
    else:
        os.mkdir('pngout')

    # Get samples from the prior, its push-forward and the observed density
    print('')
    print('Calculating the Prior push-forward ...')
    prior_samples, prior_pf_samples, obs_loc, obs_scale, prior_pf_mc_samples = get_prior_prior_pf_samples(n_samples)

    end = time.time()
    print('(BMFMC elapsed time: %fs)\n' % (end - start))
    lap = time.time()

    # Prior
    p_prior = Distribution(prior_samples, rv_name='$\lambda$', label='Prior', kde=False)

    # Prior push-forward
    p_prior_pf = Distribution(prior_pf_samples[-1, :, :], rv_name='$Q$', label='Prior-PF')

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

    end = time.time()
    print('(High-fidelity CBayes elapsed time: %fs)\n' % (end - lap))
    lap = time.time()

    # Low-fidelity and Monte Carlo comparisons
    if pf_method == 'bmfmc':

        kls = np.zeros((len(n_evals) + 1))
        kls[-1] = cbayes_post.get_prior_post_kl()

        # Create low-fidelity CBayes posteriors
        for i in range(len(n_evals)):
            print('Evaluating the low-fidelity posteriors %d / %d ...' % (i + 1, len(n_evals)))
            p_prior_pf_lf = Distribution(prior_pf_samples[i, :, :], rv_name='$Q$', label='Prior-PF-LF')
            obs_samples = np.random.randn(n_samples) * obs_scale + obs_loc
            obs_samples = np.reshape(obs_samples, (n_samples, 1))
            p_obs = Distribution(obs_samples, rv_name='$Q$', label='Observed')
            cbayes_post_lf = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf_lf)
            cbayes_post_lf.setup_posterior_and_pf()
            cbayes_post_lf.print_stats()
            kls[i] = cbayes_post_lf.get_prior_post_kl()
            if i == 0:
                cbayes_post_lf.plot_results(2)

        end = time.time()
        print('(Low-fidelities CBayes elapsed time: %fs)\n' % (end - lap))
        lap = time.time()

        # Monte Carlo comparison
        p_prior_pf_mc = Distribution(prior_pf_mc_samples, rv_name='$Q$', label='Prior-PF-MC')
        obs_samples = np.random.randn(n_samples) * obs_scale + obs_loc
        obs_samples = np.reshape(obs_samples, (n_samples, 1))
        p_obs = Distribution(obs_samples, rv_name='$Q$', label='Observed')
        cbayes_post_mc = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf_mc)
        cbayes_post_mc.setup_posterior_and_pf()
        print('Evaluating the Monte Carlo posterior ...')
        cbayes_post_mc.print_stats()
        mc_kl = cbayes_post_mc.get_prior_post_kl()
        cbayes_post_mc.plot_results(3)

        end = time.time()
        print('(Monte Carlo CBayes elapsed time: %fs)\n' % (end - lap))
        lap = time.time()

        # Plot Posterior-Prior KLs
        plt.figure(4)
        plt.plot(range(1, len(n_evals) + 2), mc_kl * np.ones((len(n_evals) + 1,)), 'k--', label='MC KL')
        plt.plot(range(1, len(n_evals) + 2), kls, '-x', label='Model KLs')
        plt.grid()
        plt.legend(loc='lower right')
        plt.xlabel('Model no.')
        plt.ylabel('KL')
        plt.title('Prior-Posterior KLs')
        plt.gcf().savefig('pngout/cbayes_prior_post_kls.png', dpi=300)

    end = time.time()
    print('(Total elapsed time: %fs)' % (end - start))


# --------------------------------------------------------------------------- #
