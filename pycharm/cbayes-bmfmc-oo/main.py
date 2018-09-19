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


# Number of lowest-fidelity samples (1e4 should be fine for 1 QoI)

n_samples = int(1e4)

# Forward models:
#   - lambda_p
#   - ellptic_pde / elliptic_pde_2d / elliptic_pde_3d
#   - ellptic_pde_ml / elliptic_pde_ml_2d / elliptic_pde_ml_3d
#   - ellptic_pde_ml_fixed / elliptic_pde_ml_fixed_2d / elliptic_pde_ml_fixed_3d
#   - ode_pp /  ode_pp_2d

model = 'elliptic_pde_3d'

# Push-forward methods:
#   - mc
#   - bmfmc

pf_method = 'bmfmc'

# ----------------------------------------------------Config - BMFMC -------- #


# Number of model evaluations in increasing fidelity (the lowest-fidelity model will always have n_samples evals)
# Only lambda_p and ode_pp support more than one (i.e. arbitrary many) low-fidelity level
# The number of models will thus be len(n_evals) + 1

n_evals = [100, 50, 20]
n_evals = [6**3]

# Training set selection strategies:
#   - support_covering / support_covering_adaptive
#   - sampling / sampling_adaptive
#   - fixed

training_set_strategy = 'sampling'

# Regression model types
#   - gaussian_process / decoupled_gaussian_processes
#   - heteroscedastic_gaussian_process / decoupled_heteroscedastic_gaussian_process

regression_type = 'decoupled_gaussian_processes'


# ---------------------------------------------------------- Todos ---------- #


# Framework
# todo: (!) BMFMC CDF and density estimation for more than 1 QoI
# todo: (!) enhance plotting: https://www.safaribooksonline.com/library/view/python-data-science/9781491912126/ch04.html
# todo: (!) think about the case where one has several lowest-fidelity levels (for 1 QoI, this implies a regression model with two inputs and one output)

# Distributions
# todo: (!) implement transformations of random variables to operate in unconstrained probability space only

# Regression
# todo: (!!) Do a covariance / correlation check before choosing shared or separate kernels for the GPs
# todo: (!) better regression for multiple QoIs (multi-output GPs would be an option)
# todo: (!) GPs with non-Gaussian noise for asymmetric correlations
# todo: (!) PyMC: https://docs.pymc.io/notebooks/GP-MeansAndCovs.html
# todo: (!) GPy: http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb
# Check: Kernels for Vector-Valued Functions: a Review

# Adaptive training
# todo: (!) different options how to choose support covering samples: uniform grid / LHS / sparse grids


# ------------------------------------------------- Models & Methods -------- #


def get_prior_prior_pf_samples(n_samples):
    prior_samples = prior_pf_samples = obs_loc = obs_scale = prior_pf_mc_samples = mc_model = n_qoi = None

    # Check push forward method
    if pf_method not in ['mc', 'bmfmc']:
        print('Unknown push-forward method: %r' % pf_method)
        exit()

    if model == 'lambda_p':

        n_qoi = 1
        obs_loc = [0.25]
        obs_scale = [0.1]
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

    elif model in ['elliptic_pde', 'elliptic_pde_2d', 'elliptic_pde_3d']:

        # Setup and load data
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

            if len(n_evals) > 1:
                print('elliptic_pde only supports 2 levels of fidelity.')
                exit()

            # Construct low-fi model
            X_train = lam[0:round(split * lam.shape[0])]
            y_train = qvals[0:round(split * lam.shape[0])]
            lf_samples = elliptic_pde.construct_lowfi_model_and_get_samples(X_train=X_train, y_train=y_train,
                                                                            X_test=prior_samples)
            # Add bias
            lf_samples[:, 0] = np.sin(lf_samples[:, 0])
            if np.shape(lf_samples)[1] > 1:
                lf_samples[:, 1:] = 1.5 * np.sin(lf_samples[:, 1:])

            lf_model = Model(eval_fun=lambda x: elliptic_pde.find_xy_pair(x, prior_samples, lf_samples),
                             rv_samples=prior_samples, rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                             rv_name='$q$', label='Low-fidelity')

            # High-fi model samples
            hf_samples = prior_pf_samples[indices, :]

            hf_model = Model(eval_fun=lambda x: elliptic_pde.find_xy_pair(x, prior_samples, hf_samples),
                             n_evals=n_evals[-1], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            models = [lf_model, hf_model]

    elif model in ['elliptic_pde_ml', 'elliptic_pde_ml_2d', 'elliptic_pde_ml_3d']:

        # Setup and load data
        if model == 'elliptic_pde_ml':
            n_qoi = 1
            obs_loc = [0.7]
            obs_scale = [0.01]

        elif model == 'elliptic_pde_ml_2d':
            n_qoi = 2
            obs_loc = [0.7, 0.1]
            obs_scale = [0.01, 0.01]

        elif model == 'elliptic_pde_ml_3d':
            n_qoi = 3
            obs_loc = [0.71, 0.12, 0.48]
            obs_scale = [0.01, 0.01, 0.01]

        h = 160 / 2 ** len(n_evals)
        prior_pf_samples = elliptic_pde_ml.load_ml_data(h=h, n_models=len(n_evals) + 1)
        prior_samples = np.reshape(range(n_samples), (n_samples, 1))  # we only need some id here

        if pf_method == 'mc':

            # Monte Carlo reference
            prior_pf_samples = prior_pf_samples[-1][:, 0:n_qoi]
            prior_pf_samples = np.reshape(prior_pf_samples, (1, n_samples, n_qoi))
            prior_pf_mc_samples = prior_pf_samples

        elif pf_method == 'bmfmc':

            if len(n_evals) > 4:
                print('elliptic_pde only supports 5 levels of fidelity.')
                exit()

            # Create a low-fidelity model
            samples = prior_pf_samples[0][:, 0:n_qoi]
            # samples = np.sin(samples)  # add a bias
            lf_model = Model(
                eval_fun=lambda x, samples=samples: elliptic_pde_ml.find_xy_pair(x, prior_samples, samples),
                rv_samples=prior_samples, rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                rv_name='$q_0$', label='Low-fidelity')

            # Create a high-fidelity model
            samples = prior_pf_samples[-1][:, 0:n_qoi]
            hf_model = Model(
                eval_fun=lambda x, samples=samples: elliptic_pde_ml.find_xy_pair(x, prior_samples, samples),
                n_evals=n_evals[-1], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            if len(n_evals) == 2:
                # Create a mid fidelity model
                samples = prior_pf_samples[1][:, 0:n_qoi]
                mf_model = Model(
                    eval_fun=lambda x, samples=samples: elliptic_pde_ml.find_xy_pair(x, prior_samples, samples),
                    n_evals=n_evals[0], n_qoi=n_qoi, rv_name='$q_m$', label='Mid-fidelity')
                models = [lf_model, mf_model, hf_model]

            elif len(n_evals) > 2:
                models = [lf_model]
                for i in range(len(n_evals) - 1):
                    samples = prior_pf_samples[i + 1][:, 0:n_qoi]
                    models.append(
                        Model(
                            eval_fun=lambda x, samples=samples: elliptic_pde_ml.find_xy_pair(x, prior_samples, samples),
                            n_evals=n_evals[i], n_qoi=n_qoi, rv_name='$q_%d$' % int(i + 1),
                            label='Mid-%d-fidelity' % int(i + 1)))
                models.append(hf_model)

            elif len(n_evals) == 1:
                models = [lf_model, hf_model]

            else:
                print('Unsupported number of models for elliptic_pde_ml.')
                exit()

    elif model in ['elliptic_pde_ml_fixed', 'elliptic_pde_ml_fixed_2d', 'elliptic_pde_ml_fixed_3d']:

        # Setup and load data
        if model == 'elliptic_pde_ml_fixed':
            n_qoi = 1
            obs_loc = [0.7]
            obs_scale = [0.01]

        elif model == 'elliptic_pde_ml_fixed_2d':
            n_qoi = 2
            obs_loc = [0.7, 0.1]
            obs_scale = [0.01, 0.01]

        elif model == 'elliptic_pde_ml_fixed_3d':
            n_qoi = 3
            obs_loc = [0.71, 0.12, 0.48]
            obs_scale = [0.01, 0.01, 0.01]

        h = 160 / 2 ** len(n_evals)
        prior_pf_samples = elliptic_pde_ml_fixed.load_data(h=h, n_evals=n_evals)
        prior_samples = np.reshape(range(n_samples), (n_samples, 1))  # we only need some id here

        if len(n_evals) > 4:
            print('elliptic_pde only supports 5 levels of fidelity.')
            exit()

        # Create a MC model
        mc_model = Model(
            eval_fun=None,
            rv_samples=prior_samples, rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
            rv_name='$Q$', label='MC reference')
        mc_model.set_model_evals(elliptic_pde_ml_fixed.load_mc_reference()[:, 0:n_qoi])

        # Create a low-fidelity model
        lf_model = Model(
            eval_fun=None,
            rv_samples=prior_samples, rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
            rv_name='$q_0$', label='Low-fidelity')
        lf_model.set_model_evals(prior_pf_samples[0][:, 0:n_qoi])

        # Create a high-fidelity model
        hf_model = Model(
            eval_fun=None,
            rv_samples=prior_samples[0:n_evals[-1], :], rv_samples_pred=prior_samples, n_evals=n_evals[-1], n_qoi=n_qoi,
            rv_name='$Q$', label='High-fidelity')
        hf_model.set_model_evals(prior_pf_samples[-1][:, 0:n_qoi])

        if len(n_evals) == 2:
            # Create a mid fidelity model
            mf_model = Model(
                eval_fun=None,
                rv_samples=prior_samples[0:n_evals[0], :], rv_samples_pred=prior_samples, n_evals=n_evals[0],
                n_qoi=n_qoi, rv_name='$q_m$', label='Mid-fidelity')
            mf_model.set_model_evals(prior_pf_samples[1][:, 0:n_qoi])
            models = [lf_model, mf_model, hf_model]

        elif len(n_evals) > 2:
            models = [lf_model]
            for i in range(len(n_evals) - 1):
                models.append(
                    Model(
                        eval_fun=None,
                        rv_samples=prior_samples[0:n_evals[i], :], rv_samples_pred=prior_samples, n_evals=n_evals[i],
                        n_qoi=n_qoi, rv_name='$q_%d$' % int(i + 1),
                        label='Mid-%d-fidelity' % int(i + 1)))
                models[i + 1].set_model_evals(prior_pf_samples[i + 1][:, 0:n_qoi])
            models.append(hf_model)

        elif len(n_evals) == 1:
            models = [lf_model, hf_model]

        else:
            print('Unsupported number of models for elliptic_pde_ml_fixed.')
            exit()

    elif model == 'ode_pp':

        n_qoi = 1
        obs_loc = [2.5]
        obs_scale = [0.1]
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

    elif model == 'ode_pp_2d':

        n_qoi = 2
        obs_loc = [2.5, 4.5]
        obs_scale = [0.25, 0.3]
        prior_samples = ode_pp.get_prior_samples(n_samples)

        # Model settings
        u0 = np.array([5, 1])
        finalt = 1.0
        dt_hf = 0.5

        hf_settings = ode_pp.Settings(finalt=finalt, dt=dt_hf, u0=u0)

        if pf_method == 'mc':

            # Create the high-fidelity model
            hf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, hf_settings)[:, -1], rv_samples=prior_samples,
                             n_evals=n_samples, n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            # Brute force Monte Carlo
            prior_pf_samples = hf_model.evaluate()
            prior_pf_samples = np.reshape(prior_pf_samples,
                                          (1, n_samples, n_qoi))
            prior_pf_mc_samples = prior_pf_samples

        elif pf_method == 'bmfmc':

            # Create a low-fidelity model
            dt_lf = 1.0
            lf_settings = ode_pp.Settings(finalt=finalt, dt=dt_lf, u0=u0)
            lf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, lf_settings)[:, -1] + 1.0,
                             rv_samples=prior_samples,
                             rv_samples_pred=prior_samples, n_evals=n_samples, n_qoi=n_qoi,
                             rv_name='$q_0$', label='Low-fidelity')

            # Create a high-fidelity model
            hf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, hf_settings)[:, -1], n_evals=n_evals[-1],
                             n_qoi=n_qoi,
                             rv_name='$Q$', label='High-fidelity')

            if len(n_evals) == 2:
                # Create a mid fidelity model
                dt_mf = 0.7
                mf_settings = ode_pp.Settings(finalt=finalt, dt=dt_mf, u0=u0)
                mf_model = Model(eval_fun=lambda x: ode_pp.ode_pp(x, mf_settings)[:, -1], n_evals=n_evals[0],
                                 n_qoi=n_qoi, rv_name='$q_m$', label='Mid-fidelity')
                models = [lf_model, mf_model, hf_model]

            elif len(n_evals) > 2:
                models = [lf_model]
                dts = np.linspace(dt_lf, dt_hf, len(n_evals) + 2)
                for i in range(len(n_evals) - 1):
                    settings = ode_pp.Settings(finalt=finalt, dt=dts[i + 1], u0=u0)
                    models.append(Model(eval_fun=lambda x, settings=settings: ode_pp.ode_pp(x, settings)[:, -1],
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
        # # print(bmfmc.calculate_bmfmc_density_expectation_estimator_variance(val=0.7))
        #
        # # Probabilities
        # print(bmfmc.calculate_bmfmc_expectation(fun=lambda x: x < 0.7))
        # print(bmfmc.calculate_bmfmc_expectation_estimator_variance(fun=lambda x: x < 0.7))
        #
        # # BMFMC mean estimator variance via Gaussian assumption and general expectation variance
        # print(bmfmc.calculate_bmfmc_mean_estimator_variance())
        # print(bmfmc.calculate_bmfmc_expectation_estimator_variance())
        #
        # # BMFMC mean and variance via general expectation
        # mean = bmfmc.calculate_bmfmc_expectation()
        # print(mean)
        # print(bmfmc.calculate_bmfmc_expectation(fun=lambda x: (x - mean)**2))
        #
        # # BMFMC mean and variance via samples
        # print(np.mean(bmfmc.models[-1].model_evals_pred, axis=0))
        # print(np.var(bmfmc.models[-1].model_evals_pred, axis=0))
        #
        # n_vals = 50
        # cdf_mean, y_range = bmfmc.calculate_bmfmc_cdf(n_vals=n_vals)
        # cdf_std, _ = bmfmc.calculate_bmfmc_cdf_estimator_variance(n_vals=n_vals)
        #
        # plt.figure()
        # plt.plot(cdf_mean)
        # plt.plot(cdf_mean + 1.96 * cdf_std)
        # plt.plot(cdf_mean - 1.96 * cdf_std)
        #
        # y_diffs = np.diff(y_range)
        # p_std = np.zeros((y_diffs.shape[0], 1))
        # p_std_m = np.zeros((y_diffs.shape[0], 1))
        # cdf_std_m = cdf_mean - 1.96*cdf_std
        # cdf_std_p = cdf_mean + 1.96 * cdf_std
        # p_std_p = np.zeros((y_diffs.shape[0], 1))
        # p_mean = np.zeros((y_diffs.shape[0], 1))
        # for i in range(y_diffs.shape[0]):
        #     p_std[i] = (cdf_std[i + 1] - cdf_std[i]) / y_diffs[i]
        #     p_mean[i] = (cdf_mean[i + 1] - cdf_mean[i]) / y_diffs[i]
        #     p_std_m[i] = (cdf_std_m[i + 1] - cdf_std_m[i]) / y_diffs[i]
        #     p_std_p[i] = (cdf_std_p[i + 1] - cdf_std_p[i]) / y_diffs[i]
        #
        # plt.figure()
        # plt.plot(p_mean)
        # plt.plot(p_std_p)
        # plt.plot(p_std_m)
        #
        # plt.show()
        # exit()

        # TEST

        if n_qoi == 1:
            bmfmc.plot_results(mc=True)
            bmfmc.plot_regression_models()
            bmfmc.plot_joint_densities()
        elif n_qoi == 2:
            bmfmc.plot_results(mc=True)
            bmfmc.plot_regression_models()

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
    obs_samples = np.random.randn(n_samples, len(obs_scale)) * obs_scale + obs_loc
    obs_samples = np.reshape(obs_samples, (n_samples, np.shape(obs_samples)[1]))
    p_obs = Distribution(obs_samples, rv_name='$Q$', label='Observed')

    # Posterior
    print('Evaluating the posterior ...')
    cbayes_post = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf)
    cbayes_post.setup_posterior_and_pf()

    # Output
    cbayes_post.print_stats()

    # Plot
    cbayes_post.plot_results(model_tag='hf')

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
            cbayes_post_lf = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf_lf)
            cbayes_post_lf.setup_posterior_and_pf()
            cbayes_post_lf.print_stats()
            kls[i] = cbayes_post_lf.get_prior_post_kl()
            if i == 0:
                cbayes_post_lf.plot_results(model_tag='lf')

        end = time.time()
        print('(Low-fidelities CBayes elapsed time: %fs)\n' % (end - lap))
        lap = time.time()

        # Monte Carlo comparison
        p_prior_pf_mc = Distribution(prior_pf_mc_samples, rv_name='$Q$', label='Prior-PF-MC')
        cbayes_post_mc = CBayesPosterior(p_obs=p_obs, p_prior=p_prior, p_prior_pf=p_prior_pf_mc)
        cbayes_post_mc.setup_posterior_and_pf()
        print('Evaluating the Monte Carlo posterior ...')
        cbayes_post_mc.print_stats()
        mc_kl = cbayes_post_mc.get_prior_post_kl()
        cbayes_post_mc.plot_results(model_tag='mc')

        end = time.time()
        print('(Monte Carlo CBayes elapsed time: %fs)\n' % (end - lap))
        lap = time.time()

        # Plot Posterior-Prior KLs
        plt.figure()
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
