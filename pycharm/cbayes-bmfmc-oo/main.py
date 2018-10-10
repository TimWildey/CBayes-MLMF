# Standard stuff
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

# Add relevant folders to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/models')

# Model stuff
import lambda_p
import elliptic_pde
import elliptic_pde_ml
import elliptic_pde_ml_fixed
import ode_pp

# Framework stuff
from Distribution import Distribution
from Model import Model
from BMFMC import BMFMC
from CBayes import CBayesPosterior

# ------------------------------------------------- Config - General -------- #

# Random seed

np.random.seed(42)

# Forward models:
#   - lambda_p
#   - elliptic_pde / elliptic_pde_2d / elliptic_pde_3d
#   - elliptic_pde_ml / elliptic_pde_ml_2d / elliptic_pde_ml_3d
#   - elliptic_pde_ml_fixed / elliptic_pde_ml_fixed_2d / elliptic_pde_ml_fixed_3d
#   - ode_pp /  ode_pp_2d

model = 'elliptic_pde_ml_2d'

# Push-forward methods (weird name...):
#   - mc
#   - bmfmc

pf_method = 'bmfmc'

# ----------------------------------------------------Config - BMFMC -------- #


# Number of model evaluations in increasing fidelity starting with the lowest fidelity
#   - elliptic_pde supports one
#   - lambda_p supports two
#   - ode_pp supports arbitrary many
#   - elliptic_pde_ml / elliptic_pde_ml_fixed support five

n_evals = [10000, 1000, 100]
n_models = len(n_evals)

# Number of samples for the Monte Carlo reference

n_mc_ref = int(5e4)

# Training set selection strategies:
#   - support_covering / support_covering_adaptive
#   - sampling / sampling_adaptive
#   - fixed

training_set_strategy = 'sampling'

# Regression model types
#   - gaussian_process
#   - decoupled_gaussian_process
#   - shared_gaussian_process
#   - heteroscedastic_gaussian_process
#   - decoupled_heteroscedastic_gaussian_process
#   - shared_heteroscedastic_gaussian_process
#   - EXPERIMENTAL: pymc_gp

regression_type = 'heteroscedastic_gaussian_process'


# ---------------------------------------------------------- Todos ---------- #


# Framework
# todo: (!!!) Check asymptotic density estimation property of BMFMC
# todo: (!) BMFMC CDF and density estimation for more than 1 QoI
# todo: (!) enhance plotting: https://www.safaribooksonline.com/library/view/python-data-science/9781491912126/ch04.html
# todo: (!) think about the case where one has several lowest-fidelity levels (for 1 QoI, this implies a regression model with two inputs and one output)

# Distributions
# todo: (!) implement transformations of random variables to operate in unconstrained probability space only

# Regression
# todo: (!) better regression for multiple QoIs (multi-output GPs would be an option)
# todo: (!) GPs with non-Gaussian noise
# todo: (!) PyMC: https://docs.pymc.io/notebooks/GP-MeansAndCovs.html
# todo: (!) GPy: http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb

# Adaptive training
# todo: (!) different options how to choose support covering samples: uniform grid / LHS / sparse grids


# ------------------------------------------------- Models & Methods -------- #


def get_prior_prior_pf_samples():
    prior_samples = prior_pf_samples = obs_loc = obs_scale = prior_pf_mc_samples = mc_model = n_qoi = None

    # Check push forward method
    if pf_method not in ['mc', 'bmfmc']:
        print('Unknown push-forward method: %r' % pf_method)
        exit()

    if model == 'lambda_p':

        n_qoi = 1
        obs_loc = [0.25]
        obs_scale = [0.1]
        prior_samples = lambda_p.get_prior_samples(n_mc_ref)

        # Create the Monte Carlo reference
        mc_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, 5), rv_samples=prior_samples,
                         rv_samples_pred=prior_samples, n_evals=n_mc_ref, n_qoi=n_qoi, rv_name='$Q$',
                         label='Monte Carlo reference')

        if pf_method == 'mc':

            # Brute force Monte Carlo
            prior_pf_samples = mc_model.evaluate()
            prior_pf_samples = np.reshape(prior_pf_samples, (1, n_mc_ref, np.shape(prior_pf_samples)[1]))
            prior_pf_mc_samples = prior_pf_samples

        elif pf_method == 'bmfmc':

            # Create a low-fidelity model
            lf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, 1), rv_samples=prior_samples[:n_evals[0]],
                             rv_samples_pred=prior_samples[:n_evals[0]], n_evals=n_evals[0], n_qoi=n_qoi,
                             rv_name='$q_0$', label='Low-fidelity')

            # Create a high-fidelity model
            hf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, 5), n_evals=n_evals[-1],
                             n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            if n_models == 2:
                models = [lf_model, hf_model]

            # Create a mid fidelity model
            elif n_models == 3:

                mf_model = Model(eval_fun=lambda x: lambda_p.lambda_p(x, 3), n_evals=n_evals[1], n_qoi=n_qoi,
                                 rv_name='$q_1$', label='Mid-fidelity')
                models = [lf_model, mf_model, hf_model]

            else:
                print('Unsupported number of models (%d) for lambda_p.' % n_models)
                exit()

    elif model in ['elliptic_pde', 'elliptic_pde_2d', 'elliptic_pde_3d']:

        # Setup and load data
        lam = qvals = None
        if model == 'elliptic_pde':
            n_qoi = 1
            obs_loc = [0.7]
            obs_scale = [0.01]
            lam, qvals = elliptic_pde.load_data()

        elif model == 'elliptic_pde_2d':
            n_qoi = 2
            obs_loc = [0.7, 0.1]
            obs_scale = [0.01, 0.01]
            lam, qvals = elliptic_pde.load_data_2d()

        elif model == 'elliptic_pde_3d':
            n_qoi = 3
            obs_loc = [0.71, 0.12, 0.48]
            obs_scale = [0.01, 0.01, 0.01]
            lam, qvals = elliptic_pde.load_data_3d()

        # Data split for surrogate model creation
        split = 0.05

        # Remaining data
        prior_samples = lam[round(split * lam.shape[0]) + 1:, :]
        prior_pf_samples = qvals[round(split * qvals.shape[0]) + 1:, :]

        # Choose at most the remaining number of samples
        n_mc_ref_new = min(round((1 - split) * len(qvals)) - 1, n_mc_ref)
        indices = np.random.choice(range(prior_pf_samples.shape[0]), size=n_mc_ref_new, replace=False)
        prior_samples = prior_samples[indices, :]
        prior_pf_samples = prior_pf_samples[indices, :]

        mc_model = Model(eval_fun=None, rv_samples=prior_samples, rv_samples_pred=prior_samples,
                         n_evals=n_mc_ref_new, n_qoi=n_qoi, rv_name='$Q$', label='Monte Carlo reference')
        mc_model.set_model_evals(prior_pf_samples)

        if pf_method == 'mc':

            # Monte Carlo reference
            prior_pf_samples = np.reshape(prior_pf_samples, (1, n_mc_ref_new, np.shape(prior_pf_samples)[1]))
            prior_pf_mc_samples = prior_pf_samples

        elif pf_method == 'bmfmc':

            if n_models > 2:
                print('elliptic_pde only supports 2 levels of fidelity.')
                exit()

            # Construct low-fi model
            X_train = lam[0:round(split * lam.shape[0])]
            y_train = qvals[0:round(split * lam.shape[0])]
            lf_prior_samples = prior_samples[:n_evals[0]]
            lf_samples = elliptic_pde.construct_lowfi_model_and_get_samples(X_train=X_train, y_train=y_train,
                                                                            X_test=lf_prior_samples)
            # Add bias
            lf_samples[:, 0] = np.sin(lf_samples[:, 0])
            if np.shape(lf_samples)[1] > 1:
                lf_samples[:, 1:] = 1.5 * np.sin(lf_samples[:, 1:])

            # Create low-fi model
            lf_model = Model(eval_fun=lambda x: elliptic_pde.find_xy_pair(x, lf_prior_samples, lf_samples),
                             rv_samples=lf_prior_samples, rv_samples_pred=lf_prior_samples, n_evals=n_evals[0],
                             n_qoi=n_qoi, rv_name='$q$', label='Low-fidelity')

            # Create high-fi model
            hf_samples = prior_pf_samples[:n_evals[0]]
            hf_model = Model(eval_fun=lambda x: elliptic_pde.find_xy_pair(x, lf_prior_samples, hf_samples),
                             n_evals=n_evals[-1], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            models = [lf_model, hf_model]

    elif model in ['elliptic_pde_ml', 'elliptic_pde_ml_2d', 'elliptic_pde_ml_3d']:

        # Setup and load data
        if model == 'elliptic_pde_ml':
            n_qoi = 1
            obs_loc = [0.74]
            obs_scale = [0.01]

        elif model == 'elliptic_pde_ml_2d':
            n_qoi = 2
            obs_loc = [0.74, 0.09]
            obs_scale = [0.01, 0.01]

        elif model == 'elliptic_pde_ml_3d':
            n_qoi = 3
            obs_loc = [0.74, 0.09, 0.46]
            obs_scale = [0.01, 0.01, 0.01]

        h = 160 / 2 ** (n_models - 1)
        prior_pf_samples = elliptic_pde_ml.load_data(h=h, n_models=n_models)
        prior_samples = np.reshape(range(n_mc_ref), (n_mc_ref, 1))  # we only need some id here

        mc_model = Model(eval_fun=None, rv_samples=prior_samples, rv_samples_pred=prior_samples,
                         n_evals=n_mc_ref, n_qoi=n_qoi, rv_name='$Q$', label='Monte Carlo reference')
        mc_model.set_model_evals(prior_pf_samples[-1][:n_mc_ref, 0:n_qoi])

        if pf_method == 'mc':

            # Monte Carlo reference
            prior_pf_samples = prior_pf_samples[-1][:n_mc_ref, 0:n_qoi]
            prior_pf_samples = np.reshape(prior_pf_samples, (1, n_mc_ref, n_qoi))
            prior_pf_mc_samples = prior_pf_samples

        elif pf_method == 'bmfmc':

            if n_models > 5:
                print('elliptic_pde_ml only supports 5 levels of fidelity.')
                exit()

            # Create a low-fidelity model
            samples = prior_pf_samples[0][:n_evals[0], 0:n_qoi]
            samples = samples ** 1.2  # add a bias
            lf_prior_samples = prior_samples[:n_evals[0]]
            lf_model = Model(
                eval_fun=lambda x, samples=samples: elliptic_pde_ml.find_xy_pair(x, lf_prior_samples, samples),
                rv_samples=lf_prior_samples, rv_samples_pred=lf_prior_samples, n_evals=n_evals[0],
                n_qoi=n_qoi, rv_name='$q_0$', label='Low-fidelity')

            # Create a high-fidelity model
            samples = prior_pf_samples[-1][:n_evals[0], 0:n_qoi]
            hf_model = Model(
                eval_fun=lambda x, samples=samples: elliptic_pde_ml.find_xy_pair(x, lf_prior_samples, samples),
                n_evals=n_evals[-1], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            if n_models == 3:
                # Create a mid fidelity model
                samples = prior_pf_samples[1][:n_evals[0], 0:n_qoi]
                samples = samples ** 1.1  # add a bias
                mf_model = Model(
                    eval_fun=lambda x, samples=samples: elliptic_pde_ml.find_xy_pair(x, lf_prior_samples, samples),
                    n_evals=n_evals[1], n_qoi=n_qoi, rv_name='$q_1$', label='Mid-fidelity')
                models = [lf_model, mf_model, hf_model]

            elif n_models > 3:
                models = [lf_model]
                for i in range(n_models - 2):
                    samples = prior_pf_samples[i + 1][:n_evals[0], 0:n_qoi]
                    models.append(
                        Model(
                            eval_fun=lambda x, samples=samples: elliptic_pde_ml.find_xy_pair(x, lf_prior_samples,
                                                                                             samples),
                            n_evals=n_evals[i + 1], n_qoi=n_qoi, rv_name='$q_%d$' % int(i + 1),
                            label='Mid-%d-fidelity' % int(i + 1)))
                models.append(hf_model)

            elif n_models == 2:
                models = [lf_model, hf_model]

            else:
                print('Unsupported number of models for elliptic_pde_ml.')
                exit()

    elif model in ['elliptic_pde_ml_fixed', 'elliptic_pde_ml_fixed_2d', 'elliptic_pde_ml_fixed_3d']:

        # Setup and load data
        if model == 'elliptic_pde_ml_fixed':
            n_qoi = 1
            obs_loc = [0.75]
            obs_scale = [0.01]

        elif model == 'elliptic_pde_ml_fixed_2d':
            n_qoi = 2
            obs_loc = [0.75, 0.06]
            obs_scale = [0.01, 0.01]

        elif model == 'elliptic_pde_ml_fixed_3d':
            n_qoi = 3
            obs_loc = [0.75, 0.06, 0.45]
            obs_scale = [0.01, 0.01, 0.01]

        h = 160 / 2 ** (n_models - 1)
        prior_pf_samples = elliptic_pde_ml_fixed.load_data(h=h, n_evals=n_evals)
        prior_samples = np.reshape(range(n_mc_ref), (n_mc_ref, 1))  # we only need some id here

        if len(n_evals) > 4:
            print('elliptic_pde only supports 5 levels of fidelity.')
            exit()

        # Create a MC model
        mc_model = Model(
            eval_fun=None,
            rv_samples=prior_samples, rv_samples_pred=prior_samples, n_evals=n_mc_ref, n_qoi=n_qoi,
            rv_name='$Q$', label='MC reference')
        mc_model.set_model_evals(elliptic_pde_ml_fixed.load_mc_reference()[:, 0:n_qoi])

        # Create a low-fidelity model
        lf_prior_samples = prior_samples[:n_evals[0], :]
        lf_model = Model(
            eval_fun=None,
            rv_samples=lf_prior_samples, rv_samples_pred=lf_prior_samples, n_evals=n_evals[0], n_qoi=n_qoi,
            rv_name='$q_0$', label='Low-fidelity')
        lf_model.set_model_evals(prior_pf_samples[0][:, 0:n_qoi] ** 1.5)  # add bias

        # Create a high-fidelity model
        hf_model = Model(
            eval_fun=None,
            rv_samples=lf_prior_samples[0:n_evals[-1], :], rv_samples_pred=lf_prior_samples, n_evals=n_evals[-1],
            n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')
        hf_model.set_model_evals(prior_pf_samples[-1][:, 0:n_qoi])

        if n_models == 3:
            # Create a mid fidelity model
            mf_model = Model(
                eval_fun=None,
                rv_samples=lf_prior_samples[0:n_evals[1], :], rv_samples_pred=lf_prior_samples, n_evals=n_evals[1],
                n_qoi=n_qoi, rv_name='$q_1$', label='Mid-fidelity')
            mf_model.set_model_evals(prior_pf_samples[1][:, 0:n_qoi] ** 1.2)  # add bias
            models = [lf_model, mf_model, hf_model]

        elif n_models > 3:
            models = [lf_model]
            for i in range(n_models - 2):
                models.append(
                    Model(
                        eval_fun=None,
                        rv_samples=lf_prior_samples[0:n_evals[i + 1], :], rv_samples_pred=lf_prior_samples,
                        n_evals=n_evals[i + 1], n_qoi=n_qoi, rv_name='$q_%d$' % int(i + 1),
                        label='Mid-%d-fidelity' % int(i + 1)))
                models[i + 1].set_model_evals(prior_pf_samples[i + 1][:, 0:n_qoi])
            models.append(hf_model)

        elif n_models == 2:
            models = [lf_model, hf_model]

        else:
            print('Unsupported number of models for elliptic_pde_ml_fixed.')
            exit()

    elif model == 'ode_pp':

        n_qoi = 1
        obs_loc = [2.5]
        obs_scale = [0.1]
        prior_samples = ode_pp.get_prior_samples(n_mc_ref)

        # Model settings
        u0 = np.array([5, 1])
        finalt = 1.0
        dt_hf = 0.5

        hf_settings = ode_pp.Settings(finalt=finalt, dt=dt_hf, u0=u0)

        # Create the high-fidelity model
        mc_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, hf_settings)[0, -1], rv_samples=prior_samples,
                         n_evals=n_mc_ref, n_qoi=n_qoi, rv_name='$Q$', label='Monte Carlo reference')

        if pf_method == 'mc':

            # Brute force Monte Carlo
            prior_pf_samples = mc_model.evaluate()
            prior_pf_samples = np.reshape(prior_pf_samples,
                                          (1, n_mc_ref, np.shape(prior_pf_samples)[1]))
            prior_pf_mc_samples = prior_pf_samples

        elif pf_method == 'bmfmc':

            # Create a low-fidelity model
            lf_prior_samples = prior_samples[:n_evals[0]]
            dt_lf = 1.0
            lf_settings = ode_pp.Settings(finalt=finalt, dt=dt_lf, u0=u0)
            lf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, lf_settings)[0, -1] + 1.0,  # add bias
                             rv_samples=lf_prior_samples, rv_samples_pred=lf_prior_samples, n_evals=n_evals[0],
                             n_qoi=n_qoi, rv_name='$q_0$', label='Low-fidelity')

            # Create a high-fidelity model
            hf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, hf_settings)[0, -1], n_evals=n_evals[-1], n_qoi=n_qoi,
                             rv_name='$Q$', label='High-fidelity')

            if n_models == 3:
                # Create a mid fidelity model
                dt_mf = 0.7
                mf_settings = ode_pp.Settings(finalt=finalt, dt=dt_mf, u0=u0)
                mf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, mf_settings)[0, -1], n_evals=n_evals[1],
                                 n_qoi=n_qoi, rv_name='$q_1$', label='Mid-fidelity')
                models = [lf_model, mf_model, hf_model]

            elif n_models > 3:
                models = [lf_model]
                dts = np.linspace(dt_lf, dt_hf, n_models + 1)
                for i in range(n_models - 2):
                    settings = ode_pp.Settings(finalt=finalt, dt=dts[i + 1], u0=u0)
                    models.append(Model(eval_fun=lambda x, settings=settings: ode_pp.ode_pp(x, settings)[0, -1],
                                        n_evals=n_evals[i + 1], n_qoi=n_qoi, rv_name='$q_%d$' % int(i + 1),
                                        label='Mid-%d-fidelity' % int(i + 1)))
                models.append(hf_model)

            elif n_models == 2:
                models = [lf_model, hf_model]

            else:
                print('Unsupported number of models for ode_pp.')
                exit()

    elif model == 'ode_pp_2d':

        n_qoi = 2
        obs_loc = [2.5, 4.5]
        obs_scale = [0.25, 0.3]
        prior_samples = ode_pp.get_prior_samples(n_mc_ref)

        # Model settings
        u0 = np.array([5, 1])
        finalt = 1.0
        dt_hf = 0.5

        hf_settings = ode_pp.Settings(finalt=finalt, dt=dt_hf, u0=u0)

        # Create the high-fidelity model
        mc_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, hf_settings)[:, -1], rv_samples=prior_samples,
                         n_evals=n_mc_ref, n_qoi=n_qoi, rv_name='$Q$', label='Monte Carlo reference')

        if pf_method == 'mc':

            # Brute force Monte Carlo
            prior_pf_samples = mc_model.evaluate()
            prior_pf_samples = np.reshape(prior_pf_samples, (1, n_mc_ref, n_qoi))
            prior_pf_mc_samples = prior_pf_samples

        elif pf_method == 'bmfmc':

            # Create a low-fidelity model
            lf_prior_samples = prior_samples[:n_evals[0]]
            dt_lf = 1.0
            lf_settings = ode_pp.Settings(finalt=finalt, dt=dt_lf, u0=u0)
            lf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, lf_settings)[:, -1] + 1.0,  # add bias
                             rv_samples=lf_prior_samples, rv_samples_pred=lf_prior_samples, n_evals=n_mc_ref,
                             n_qoi=n_qoi, rv_name='$q_0$', label='Low-fidelity')

            # Create a high-fidelity model
            hf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, hf_settings)[:, -1], n_evals=n_evals[-1],
                             n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            if n_models == 3:
                # Create a mid fidelity model
                dt_mf = 0.7
                mf_settings = ode_pp.Settings(finalt=finalt, dt=dt_mf, u0=u0)
                mf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, mf_settings)[:, -1], n_evals=n_evals[1],
                                 n_qoi=n_qoi, rv_name='$q_1$', label='Mid-fidelity')
                models = [lf_model, mf_model, hf_model]

            elif n_models > 3:
                models = [lf_model]
                dts = np.linspace(dt_lf, dt_hf, n_models + 1)
                for i in range(n_models - 2):
                    settings = ode_pp.Settings(finalt=finalt, dt=dts[i + 1], u0=u0)
                    models.append(Model(eval_fun=lambda x, settings=settings: ode_pp.ode_pp(x, settings)[:, -1],
                                        n_evals=n_evals[i + 1], n_qoi=n_qoi, rv_name='$q_%d$' % int(i + 1),
                                        label='Mid-%d-fidelity' % int(i + 1)))
                models.append(hf_model)

            elif n_models == 2:
                models = [lf_model, hf_model]

            else:
                print('Unsupported number of models for ode_pp.')
                exit()

    else:
        print('Unknown model: %r' % model)
        exit()

    if pf_method == 'mc':
        print('')
        print('########### MC statistics ###########')
        print('')
        print('MC mean:\t\t\t\t\t\t%s' % prior_pf_mc_samples[0, :, :].mean(axis=0))
        print('MC std:\t\t\t\t\t\t\t%s' % prior_pf_mc_samples[0, :, :].std(axis=0))
        print('')
        print('########################################')
        print('')

    if pf_method == 'bmfmc':
        # Setup BMFMC
        bmfmc = BMFMC(models=models, mc_model=mc_model,
                      training_set_strategy=training_set_strategy, regression_type=regression_type)

        # Apply BMFMC
        bmfmc.apply_bmfmc_framework()

        # Calculate Monte Carlo reference
        bmfmc.calculate_mc_reference()

        # Diagnostics
        bmfmc.print_stats(mc=True)

        # TEST

        # # Densities
        # print(bmfmc.calculate_bmfmc_density_expectation(val=0.7))

        # # Probabilities
        # fun = lambda x: (x > 0.1) & (x < 0.3)
        # print(bmfmc.calculate_bmfmc_expectation(fun=fun))
        # print(bmfmc.calculate_bmfmc_expectation_estimator_variance(fun=fun))
        # exit()

        # n_vals = 20
        # cdf_mean, y_range = bmfmc.calculate_bmfmc_cdf(n_vals=n_vals)
        # cdf_var, _ = bmfmc.calculate_bmfmc_cdf_estimator_variance(n_vals=n_vals)
        # cdf_std = np.sqrt(cdf_var)
        #
        # plt.figure()
        # plt.plot(cdf_mean, 'C0-', label='CDF')
        # plt.plot(cdf_mean + 1.96 * cdf_std, 'C1--', label='Error bounds')
        # plt.plot(cdf_mean - 1.96 * cdf_std, 'C1--')
        # plt.legend(loc='lower right')
        # plt.xlabel('$Q$')
        # plt.ylabel('Pr$[Q]$')
        # plt.title('CDF')
        # plt.gcf().savefig('pngout/bmfmc_cdf.png', dpi=300)
        # exit()

        # TEST

        bmfmc.plot_results(mc=True)
        bmfmc.plot_regression_models()
        bmfmc.plot_joint_densities()

        # Get prior push-forward samples
        prior_pf_samples = bmfmc.get_samples()
        prior_pf_mc_samples = bmfmc.get_mc_samples()

    return prior_samples, prior_pf_samples, obs_loc, obs_scale, prior_pf_mc_samples


# ------------------------------------------------------------- Main -------- #


if __name__ == '__main__':

    # Timing
    start = time.time()

    # Get samples from the prior, its push-forward and the observed density
    print('')
    print('Calculating the Prior push-forward ...')
    prior_samples, prior_pf_samples, obs_loc, obs_scale, prior_pf_mc_samples = get_prior_prior_pf_samples()

    end = time.time()
    print('(BMFMC elapsed time: %fs)\n' % (end - start))
    lap = time.time()

    # Prior
    p_prior = Distribution(prior_samples, rv_name='$\lambda$', label='Prior', kde=False)

    # Prior push-forward
    p_prior_pf = Distribution(prior_pf_samples[-1, :, :], rv_name='$Q$', label='Prior-PF')

    # Observed density
    obs_samples = np.random.randn(n_mc_ref, len(obs_scale)) * obs_scale + obs_loc
    obs_samples = np.reshape(obs_samples, (n_mc_ref, np.shape(obs_samples)[1]))
    p_obs = Distribution(obs_samples, rv_name='$Q$', label='Observed')

    # Posterior
    print('Evaluating the posterior ...')
    cbayes_post = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf)
    cbayes_post.setup_posterior_and_pf()

    # Output
    cbayes_post.print_stats()

    # Plotting and postprocessing (a bit messy...)
    cbayes_post.plot_results(model_tag='hf')

    end = time.time()
    print('(High-fidelity CBayes elapsed time: %fs)\n' % (end - lap))
    lap = time.time()

    # Low-fidelity and Monte Carlo comparisons
    if pf_method == 'bmfmc':

        cbayes_post.plot_posterior(fignum=5, color='C%d' % (n_models - 1), label='High-fidelity')

        # Monte Carlo comparison
        lap = time.time()
        p_prior_pf_mc = Distribution(prior_pf_mc_samples, rv_name='$Q$', label='Prior-PF')
        cbayes_post_mc = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf_mc)
        cbayes_post_mc.setup_posterior_and_pf()
        print('Evaluating the Monte Carlo posterior ...')
        cbayes_post_mc.print_stats()
        mc_kl = cbayes_post_mc.get_prior_post_kl()
        cbayes_post_mc.plot_results(model_tag='mc')
        cbayes_post_mc.plot_posterior(fignum=5, color='k', linestyle='--', label='MC reference')

        end = time.time()
        print('(Monte Carlo CBayes elapsed time: %fs)\n' % (end - lap))

        kls = np.zeros(n_models)
        kls[-1] = cbayes_post.get_prior_post_kl()
        pf_kls = np.zeros(n_models)
        pf_kls[-1] = cbayes_post_mc.p_prior_pf.calculate_kl_divergence(cbayes_post.p_prior_pf)

        # Create low-fidelity CBayes posteriors
        lap = time.time()
        save_fig = False
        for i in range(n_models - 1):

            if i == n_models - 2:
                save_fig = True

            print('Evaluating the low-fidelity posteriors %d / %d ...' % (i + 1, n_models - 1))
            p_prior_pf_lf = Distribution(prior_pf_samples[i, :, :], rv_name='$Q$', label='Prior-PF')
            cbayes_post_lf = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf_lf)
            cbayes_post_lf.setup_posterior_and_pf()
            cbayes_post_lf.print_stats()

            if i == 0:
                cbayes_post_lf.plot_posterior(fignum=5, color='C%d' % i, label='Low-fidelity', save_fig=save_fig)
            elif i == 1:
                cbayes_post_lf.plot_posterior(fignum=5, color='C%d' % i, label='Mid-fidelity', save_fig=save_fig)
            else:
                cbayes_post_lf.plot_posterior(fignum=5, color='C%d' % i, label='Mid-%d-fidelity' % (i + 1),
                                              save_fig=save_fig)

            kls[i] = cbayes_post_lf.get_prior_post_kl()
            pf_kls[i] = cbayes_post_mc.p_prior_pf.calculate_kl_divergence(cbayes_post_lf.p_prior_pf)
            if i == 0:
                cbayes_post_lf.plot_results(model_tag='lf')

        end = time.time()
        print('(Low-fidelities CBayes elapsed time: %fs)\n' % (end - lap))

        # Plot Posterior-Prior KLs
        plt.figure()
        plt.plot(range(1, n_models + 1), mc_kl * np.ones((n_models,)), 'k--', label='MC reference')
        plt.plot(range(1, n_models + 1), kls, 'k-')
        for i in range(n_models):
            if i is 0:
                label = 'Low-Fidelity'
            elif i is n_models - 1:
                label = 'High-Fidelity'
            elif i is 1 and n_models is 3:
                label = 'Mid-Fidelity'
            else:
                label = 'Mid-%d-Fidelity' % (i + 1)
            plt.plot(i + 1, kls[i], 'C%do' % i, markersize=10, label=label)
        plt.grid()
        plt.legend(loc='lower right')
        plt.ylabel('KL')
        plt.xticks([])
        plt.gcf().savefig('pngout/cbayes_prior_post_kls.png', dpi=300)

        # Plot Prior-PF KLs
        plt.figure()
        plt.plot(range(1, n_models + 1), np.zeros((n_models,)), 'k--')
        plt.plot(range(1, n_models + 1), pf_kls, 'k-')
        for i in range(n_models):
            if i is 0:
                label = 'Low-Fidelity'
            elif i is n_models - 1:
                label = 'High-Fidelity'
            elif i is 1 and n_models is 3:
                label = 'Mid-Fidelity'
            else:
                label = 'Mid-%d-Fidelity' % (i + 1)
            plt.plot(i + 1, pf_kls[i], 'C%do' % i, markersize=10, label=label)
        plt.grid()
        plt.legend(loc='upper right')
        plt.ylabel('KL')
        plt.xticks([])
        plt.gcf().savefig('pngout/bmfmc_prior_pf_kls.png', dpi=300)

    # Output directory name
    outdirname = model + '_' + training_set_strategy + '_' + regression_type + '_' + str(n_mc_ref)
    for i in range(len(n_evals)):
        outdirname += '_' + str(n_evals[i])

    # Clean output folder
    if os.path.isdir('pngout/' + outdirname):
        os.system('rm -rf pngout/' + outdirname)
        os.mkdir('pngout/' + outdirname)
    else:
        os.mkdir('pngout/' + outdirname)

    # Move pngs into output folder
    os.system('mv pngout/*.png pngout/' + outdirname + '/')
    os.system('mv pngout/output.txt pngout/' + outdirname + '/')

    end = time.time()
    print('(Total elapsed time: %fs)' % (end - start))

# --------------------------------------------------------------------------- #
