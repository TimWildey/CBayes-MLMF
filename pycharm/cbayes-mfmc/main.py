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
#   - elliptic_pde
#   - elliptic_pde_2d
#   - elliptic_pde_3d

model = 'lambda_p'

# Forward UQ method:
#   - mc
#   - bmfmc

fw_uq_method = 'bmfmc'

# ----------------------------------------------------Config - BMFMC -------- #


# Number of model evaluations in increasing fidelity starting with the lowest fidelity

n_evals = [20000, 50, 25]
n_models = len(n_evals)

# Number of samples for the Monte Carlo reference

n_mc_ref = int(2e4)

# Training set selection strategies:
#   - support_covering
#   - sampling

training_set_strategy = 'support_covering'

# Regression model types
#   - gaussian_process
#   - heteroscedastic_gaussian_process
#   - decoupled_gaussian_process
#   - decoupled_heteroscedastic_gaussian_process

regression_type = 'gaussian_process'


# ------------------------------------------------- Models & Methods -------- #


def get_prior_prior_pf_samples():
    prior_samples = prior_pf_samples = obs_loc = obs_scale = prior_pf_mc_samples = mc_model = n_qoi = None

    # Check push forward method
    if fw_uq_method not in ['mc', 'bmfmc']:
        print('Unknown push-forward method: %r' % fw_uq_method)
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

        if fw_uq_method == 'mc':

            # Brute force Monte Carlo
            prior_pf_samples = mc_model.evaluate()
            prior_pf_samples = np.reshape(prior_pf_samples, (1, n_mc_ref, np.shape(prior_pf_samples)[1]))
            prior_pf_mc_samples = prior_pf_samples

        elif fw_uq_method == 'bmfmc':

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
        if model == 'elliptic_pde':
            n_qoi = 1
            obs_loc = [0.71]
            obs_scale = [0.02]

        elif model == 'elliptic_pde_2d':
            n_qoi = 2
            obs_loc = [0.71, 0.12]
            obs_scale = [0.02, 0.02]

        elif model == 'elliptic_pde_3d':
            n_qoi = 3
            obs_loc = [0.71, 0.12, 0.45]
            obs_scale = [0.02, 0.02, 0.02]

        prior_pf_samples = elliptic_pde.load_data()
        prior_samples = np.reshape(range(n_mc_ref), (n_mc_ref, 1))  # we only need some id here

        mc_model = Model(eval_fun=None, rv_samples=prior_samples, rv_samples_pred=prior_samples,
                         n_evals=n_mc_ref, n_qoi=n_qoi, rv_name='$Q$', label='Monte Carlo reference')
        mc_model.set_model_evals(prior_pf_samples[-1][:n_mc_ref, 0:n_qoi])

        if fw_uq_method == 'mc':

            # Monte Carlo reference
            prior_pf_samples = prior_pf_samples[-1][:n_mc_ref, 0:n_qoi]
            prior_pf_samples = np.reshape(prior_pf_samples, (1, n_mc_ref, n_qoi))
            prior_pf_mc_samples = prior_pf_samples

        elif fw_uq_method == 'bmfmc':

            if n_models > 3:
                print('elliptic_pde only supports up to 3 fidelity levels.')
                exit()

            # Create a low-fidelity model
            samples = prior_pf_samples[0][:n_evals[0], 0:n_qoi]
            samples = samples ** 1.2  # add a bias
            lf_prior_samples = prior_samples[:n_evals[0]]
            lf_model = Model(
                eval_fun=lambda x, samples=samples: elliptic_pde.find_xy_pair(x, lf_prior_samples, samples),
                rv_samples=lf_prior_samples, rv_samples_pred=lf_prior_samples, n_evals=n_evals[0],
                n_qoi=n_qoi, rv_name='$q_0$', label='Low-fidelity')

            # Create a high-fidelity model
            samples = prior_pf_samples[-1][:n_evals[0], 0:n_qoi]
            hf_model = Model(
                eval_fun=lambda x, samples=samples: elliptic_pde.find_xy_pair(x, lf_prior_samples, samples),
                n_evals=n_evals[-1], n_qoi=n_qoi, rv_name='$Q$', label='High-fidelity')

            if n_models == 3:
                # Create a mid fidelity model
                samples = prior_pf_samples[1][:n_evals[0], 0:n_qoi]
                samples = samples ** 1.1  # add a bias
                mf_model = Model(
                    eval_fun=lambda x, samples=samples: elliptic_pde.find_xy_pair(x, lf_prior_samples, samples),
                    n_evals=n_evals[1], n_qoi=n_qoi, rv_name='$q_1$', label='Mid-fidelity')
                models = [lf_model, mf_model, hf_model]

            elif n_models == 2:
                models = [lf_model, hf_model]

            else:
                print('Unsupported number of models (%d) for elliptic_pde.' % n_models)
                exit()

    else:
        print('Unknown model: %r' % model)
        exit()

    if fw_uq_method == 'mc':
        print('')
        print('########### MC statistics ###########')
        print('')
        print('MC mean:\t\t\t\t\t\t%s' % prior_pf_mc_samples[0, :, :].mean(axis=0))
        print('MC std:\t\t\t\t\t\t\t%s' % prior_pf_mc_samples[0, :, :].std(axis=0))
        print('')
        print('########################################')
        print('')

    if fw_uq_method == 'bmfmc':
        # Setup BMFMC
        bmfmc = BMFMC(models=models, mc_model=mc_model,
                      training_set_strategy=training_set_strategy, regression_type=regression_type)

        # Apply BMFMC
        bmfmc.apply_bmfmc_framework()

        # Calculate Monte Carlo reference
        bmfmc.calculate_mc_reference()

        # Diagnostics
        bmfmc.print_stats(mc=True)

        # Plots
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
    if fw_uq_method == 'bmfmc':

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
        plt.gcf().savefig('output/cbayes_prior_post_kls.eps', dpi=300)

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
        plt.gcf().savefig('output/bmfmc_prior_pf_kls.eps', dpi=300)

    # Output directory name
    outdirname = model + '_' + training_set_strategy + '_' + regression_type + '_' + str(n_mc_ref)
    for i in range(len(n_evals)):
        outdirname += '_' + str(n_evals[i])

    # Clean output folder
    if os.path.isdir('output/' + outdirname):
        os.system('rm -rf output/' + outdirname)
        os.mkdir('output/' + outdirname)
    else:
        os.mkdir('output/' + outdirname)

    # Move pngs into output folder
    os.system('mv output/*.eps output/' + outdirname + '/')
    os.system('mv output/output.txt output/' + outdirname + '/')

    end = time.time()
    print('(Total elapsed time: %fs)' % (end - start))

# --------------------------------------------------------------------------- #
