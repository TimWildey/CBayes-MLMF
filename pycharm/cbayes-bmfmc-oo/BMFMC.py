import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import warnings

import utils
from Model import Model
from Regression import Regression
from Distribution import Distribution


class BMFMC:

    # Class attributes:
    # - models: a list of all models for BMFMC (i.e. at least one high-fidelity and one low-fidelity model)
    # - mc_model: the Monte Carlo reference model (computed only if calculate_mc_reference() is called)
    # - n_models: number of models
    # - training_set_selection: strategy for training set selection
    # - regression_model: regression model for the conditionals p(q_l|q_l-1)

    # Constructor
    def __init__(self, models, training_set_strategy, regression_type, mc_model=None):

        self.models = models
        self.mc_model = mc_model
        self.n_models = len(models)
        self.training_set_strategy = training_set_strategy
        self.regression_type = regression_type
        self.regression_models = []
        self.adaptive = True
        self.adaptive_tol = 2.0e-3

    # Main method in bmfmc: evaluate, regress, predict
    def apply_bmfmc_framework(self):

        for i in range(self.n_models - 1):

            lf_model = self.models[i]
            hf_model = self.models[i + 1]
            self.adaptive = True

            if self.n_models == 2:
                print('')
                print('Training the conditional model ...')
            else:
                print('')
                print('Creating conditional model %d / %d ...' % (i + 1, self.n_models - 1))

            # 1) Evaluate the lowest-fidelity model
            if i == 0:
                lf_model.evaluate()
                lf_model.set_model_evals_pred(lf_model.model_evals)
                lf_model.create_distribution()

            # Add samples adaptively until convergence
            previous_dist = lf_model.distribution
            x_train = None
            adaptive_run = 0

            while self.adaptive:
                adaptive_run += 1

                # 2) Select lower-fidelity model evaluation points and evaluate the next higher-fidelity model
                if x_train is None:
                    x_train = self.create_training_set(lf_model=lf_model, hf_model=hf_model, id=i)
                else:
                    x_train = np.append(x_train, self.create_training_set(lf_model=lf_model, hf_model=hf_model, id=i),
                                        axis=0)
                y_train = hf_model.evaluate()

                # 3) Fit a regression model to approximate p(q_l|q_l-1), predict q_l|q_l-1 at all low-fidelity samples,
                #    generate low-fidelity samples from the predictions and create a distribution
                hf_model_evals_pred = self.build_regression_predict_and_sample(x_train=x_train, y_train=y_train,
                                                                               x_pred=lf_model.model_evals_pred,
                                                                               id=i)
                hf_model.set_rv_samples_pred(lf_model.rv_samples_pred)
                hf_model.set_model_evals_pred(hf_model_evals_pred.reshape((self.models[0].n_evals, hf_model.n_qoi)))
                hf_model.create_distribution()

                # 4) Check convergence
                if self.adaptive:
                    n_evals = np.shape(hf_model.rv_samples)[0]
                    self.check_adaptive_convergence(this_dist=hf_model.distribution, previous_dist=previous_dist,
                                                    adaptive_run=adaptive_run, n_evals=n_evals)
                    previous_dist = hf_model.distribution

        return self.models[-1].model_evals_pred

    # Check for convergence in terms of the KL to the previous model with less samples
    def check_adaptive_convergence(self, this_dist, previous_dist, adaptive_run, n_evals):

        kl = this_dist.calculate_kl_divergence(previous_dist)

        print('')
        print('Adaptive run %d, KL: %f' % (adaptive_run, kl))

        if kl <= self.adaptive_tol:
            self.adaptive = False
            print('Converged after %d model evaluations.' % n_evals)
        elif adaptive_run >= 20:
            print('No convergence after 20 runs... aborting.')
            exit()

    # Create the training set for the regression
    def create_training_set(self, lf_model, hf_model, id):

        regression_model = Regression(regression_type=self.regression_type,
                                      training_set_strategy=self.training_set_strategy)

        x_train, self.adaptive = regression_model.create_training_set(lf_model=lf_model, hf_model=hf_model, id=id,
                                                                      regression_models=self.regression_models)

        # Save regression model
        if len(self.regression_models) == id:
            self.regression_models.append(regression_model)
        elif len(self.regression_models) == id + 1:
            self.regression_models[id] = regression_model
        else:
            print('This is not supposed to happen. Something went very wrong.')
            exit()

        return x_train

    # Create the regression model and make high-fidelity predictions
    def build_regression_predict_and_sample(self, x_train, y_train, x_pred, id):

        self.regression_models[id].x_train = x_train
        self.regression_models[id].y_train = y_train
        self.regression_models[id].x_pred = x_pred

        return self.regression_models[id].build_regression_predict_and_sample()

    # Obtain the predicted highest-fidelity samples
    def get_high_fidelity_samples(self):

        return self.models[-1].model_evals_pred

    # Obtain the lowest-fidelity samples
    def get_low_fidelity_samples(self):

        return self.models[0].model_evals_pred

    # Obtain the Monte Carlo reference samples
    def get_mc_samples(self):

        if self.mc_model is not None:

            return self.mc_model.model_evals_pred

        else:
            print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
            exit()

    # Obtain samples from all models (exluding the MC reference)
    def get_samples(self):

        samples = np.zeros((self.n_models, self.models[0].n_samples, self.models[0].n_qoi))
        for i in range(self.n_models):
            samples[i, :, :] = self.models[i].model_evals_pred

        return samples

    # Calculate a Monte Carlo reference solution
    def calculate_mc_reference(self):

        if self.mc_model is None:
            self.mc_model = Model(eval_fun=self.models[-1].eval_fun, n_evals=self.models[0].n_evals,
                                  n_qoi=self.models[-1].n_qoi, rv_samples=self.models[0].rv_samples,
                                  rv_samples_pred=self.models[0].rv_samples, label='MC reference',
                                  rv_name=self.models[-1].rv_name)

        if self.mc_model.model_evals is None:
            self.mc_model.evaluate()

        self.mc_model.set_model_evals_pred(self.mc_model.model_evals)
        self.mc_model.create_distribution()

    # Calculate BMFMC estimator variance of the distribution mean
    def calculate_bmfmc_mean_estimator_variance(self):

        if self.n_models == 2 and self.models[0].n_qoi == 1:
            regression_model = self.regression_models[0]
            sigma = regression_model.sigma
            return np.mean(sigma ** 2, axis=0)

        else:
            return self.calculate_bmfmc_expectation_estimator_variance()

    # Calculate the CDF
    def calculate_bmfmc_cdf(self, n_vals=10):

        if self.models[0].n_qoi > 1:
            print('This only works for one QoI so far...')
            exit()

        # Get regression model
        regression_model = self.regression_models[-1]
        mu = regression_model.mu
        sigma = regression_model.sigma

        # Approximate CDF
        min = np.percentile(self.models[-1].distribution.samples, 1)
        max = np.percentile(self.models[-1].distribution.samples, 99)
        y_range = np.linspace(min, max, n_vals)
        n_lf = self.models[0].n_samples
        cdf_samples = np.zeros((n_lf, y_range.shape[0]))
        print('')
        for i in range(y_range.shape[0]):
            print('CDF means %d / %d' % (i + 1, y_range.shape[0]))
            for j in range(n_lf):
                cdf_samples[j, i] = stats.norm.cdf((y_range[i] - mu[j]) / (sigma[j] + 1e-15))

        return np.mean(cdf_samples, 0), y_range

    # Calculate CDF estimator error bars
    def calculate_bmfmc_cdf_estimator_variance(self, n_vals=10):

        if self.models[0].n_qoi > 1:
            print('This only works for one QoI so far...')
            exit()

        min = np.percentile(self.models[-1].distribution.samples, 1)
        max = np.percentile(self.models[-1].distribution.samples, 99)
        y_range = np.linspace(min, max, n_vals)

        if self.n_models == 2:

            # Get regression model
            regression_model = self.regression_models[0]
            mu = regression_model.mu
            sigma = regression_model.sigma

            # Approximate CDF
            n_lf = self.models[0].n_samples
            cdf_var = np.zeros((n_lf, y_range.shape[0]))

            print('')
            for i in range(y_range.shape[0]):
                print('CDF errors %d / %d' % (i + 1, y_range.shape[0]))
                for j in range(n_lf):
                    cdf_var[j, i] = stats.norm.cdf((y_range[i] - mu[j]) / (sigma[j] + 1e-15)) - stats.norm.cdf(
                        (y_range[i] - mu[j]) / (sigma[j] + 1e-15)) ** 2

            return np.mean(cdf_var, 0), y_range

        else:
            cdf_var = np.zeros((y_range.shape[0],))
            print('')
            for i in range(y_range.shape[0]):
                print('CDF errors %d / %d' % (i + 1, y_range.shape[0]))
                cdf_var[i] = self.calculate_bmfmc_expectation_estimator_variance(lambda x: (x < y_range[i]))

            return cdf_var, y_range

    # Calculate the expected value of an arbitrary function
    def calculate_bmfmc_expectation(self, fun=lambda x: x):

        samples = self.models[-1].model_evals_pred
        exp_samples = fun(samples)

        return np.mean(exp_samples, 0)

    # Estimate the BMFMC estimator variance of some arbitrary expectation value
    def calculate_bmfmc_expectation_estimator_variance(self, fun=lambda x: x):

        regression_model = self.regression_models[0]
        mu = regression_model.mu
        sigma = regression_model.sigma
        n_qoi = self.models[0].n_qoi
        n_lf = np.shape(mu)[0]
        n_lf_i = 50

        exp_var_samples = np.zeros((n_lf, n_qoi))

        # Loop over all low-fidelity samples
        for i in range(n_lf):
            if self.n_models > 2 and i % 50 == 0:
                print('\rCalculating errors... (%d / %d) ' % (i, n_lf), end='')
            elif self.n_models > 2 and i == n_lf - 1:
                print('\r', end='')

            mc_mean = 1.0
            mc_error = 1.0
            j = 0
            j_max = 10
            tol = 0.05
            samples = None

            # Iterate until the MC error is acceptable
            while ((np.abs(mc_mean) + mc_error + 1e-15) / (np.abs(mc_mean) + 1e-15) > (1 + tol)).all() and j < j_max:

                # Create samples given q_i
                samples_j = np.random.randn(n_lf_i, n_qoi) * sigma[i] + mu[i]

                # Propagate samples through the regression models
                for k in range(1, self.n_models - 1):
                    samples_j = self.regression_models[k].predict_and_sample(x_pred=samples_j)

                # Apply the arbitrary function
                samples_j = fun(samples_j)

                if j > 0:
                    samples = np.vstack([samples, samples_j])
                else:
                    samples = samples_j

                # Obtain variance estimate
                exp_var_samples[i] = np.var(samples, axis=0)

                # Calculate MC error
                mc_mean = np.mean(samples, axis=0)
                mc_error = np.sqrt(exp_var_samples[i] / np.shape(samples)[0])
                j += 1

        # Return averaged variance over all low-fidelity samples
        return np.mean(exp_var_samples, 0)

    # Estimate the probability density at some point
    def calculate_bmfmc_density_expectation(self, val):

        if self.models[0].n_qoi > 1:
            print('This only works for one QoI so far...')
            exit()

        # Get regression model
        regression_model = self.regression_models[-1]
        mu = regression_model.mu
        sigma = regression_model.sigma

        n_lf = self.models[0].n_samples
        pdf_samples = np.zeros((n_lf, 1))

        # Generate samples and average
        for i in range(n_lf):
            pdf_samples[i] = stats.norm(loc=mu[i], scale=sigma[i]).pdf(val)

        return np.mean(pdf_samples, 0)

    # Print some stats
    def print_stats(self, mc=False):

        print('')
        print('########### BMFMC statistics ###########')
        print('')
        if mc and self.mc_model != 0:
            print('MC mean:\t\t\t\t\t\t%s' % self.mc_model.distribution.mean())
            print('MC std:\t\t\t\t\t\t\t%s' % self.mc_model.distribution.std())
            # print('MC skew:\t\t\t\t\t\t%s' % self.mc_model.distribution.skew())
            # print('MC kurt:\t\t\t\t\t\t%s' % self.mc_model.distribution.kurt())
            print('')
            print('MC-BMFMC KL:\t\t\t\t\t%f' % self.mc_model.distribution.calculate_kl_divergence(
                self.models[-1].distribution))
        elif mc and self.mc_model == 0:
            print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
            exit()
        print('')
        print('Low-fidelity mean:\t\t\t\t%s' % self.models[0].distribution.mean())
        print('Low-fidelity std:\t\t\t\t%s' % self.models[0].distribution.std())
        print('')
        for i in range(self.n_models - 2):
            print('Mid-%d-fidelity mean:\t\t\t%s' % (int(i + 1), self.models[i + 1].distribution.mean()))
            print('Mid-%d-fidelity std:\t\t\t\t%s' % (int(i + 1), self.models[i + 1].distribution.std()))
            print('')
            kl = self.models[i].distribution.calculate_kl_divergence(self.models[i + 1].distribution)
            print('Relative information gain:\t\t%f' % kl)
            print('')
        print('High-fidelity mean:\t\t\t\t%s' % self.models[-1].distribution.mean())
        print('High-fidelity std:\t\t\t\t%s' % self.models[-1].distribution.std())
        # print('High-fidelity skew:\t\t\t\t%s' % self.models[-1].distribution.skew())
        # print('High-fidelity kurt:\t\t\t\t%s' % self.models[-1].distribution.kurt())
        print('')
        bmfmc_mean = self.calculate_bmfmc_expectation(fun=lambda x: x)
        bmfmc_mean_error = np.sqrt(self.calculate_bmfmc_mean_estimator_variance())
        print('BMFMC mean estimator abs err:\t%s' % bmfmc_mean_error)
        print('BMFMC mean estimator rel err:\t%s' % (bmfmc_mean_error / np.abs(bmfmc_mean)))
        bmfmc_std = self.models[-1].distribution.std()
        bmfmc_var = self.models[-1].distribution.var()
        bmfmc_var_error = np.sqrt(
            self.calculate_bmfmc_expectation_estimator_variance(fun=lambda x: (x - bmfmc_mean) ** 2))
        bmfmc_std_error = np.sqrt(bmfmc_var + bmfmc_var_error) - bmfmc_std
        print('BMFMC std estimator abs err:\t%s' % bmfmc_std_error)
        print('BMFMC std estimator rel err:\t%s' % (bmfmc_std_error / bmfmc_std))
        # bmfmc_skew = self.models[-1].distribution.skew()
        # bmfmc_skew_error = np.sqrt(
        #     self.calculate_bmfmc_expectation_estimator_variance(fun=lambda x: ((x - bmfmc_mean) / bmfmc_std) ** 3))
        # print('BMFMC skew estimator abs err:\t%s' % bmfmc_skew_error)
        # print('BMFMC skew estimator rel err:\t%s' % (bmfmc_skew_error / np.abs(bmfmc_skew)))
        # bmfmc_kurt = self.models[-1].distribution.kurt()
        # bmfmc_kurt_error = np.sqrt(
        #     self.calculate_bmfmc_expectation_estimator_variance(fun=lambda x: ((x - bmfmc_mean) / bmfmc_std) ** 4))
        # print('BMFMC kurt estimator abs err:\t%s' % bmfmc_kurt_error)
        # print('BMFMC kurt estimator rel err:\t%s' % (bmfmc_kurt_error / bmfmc_kurt))
        print('')
        kl = self.models[-2].distribution.calculate_kl_divergence(self.models[-1].distribution)
        print('Relative information gain:\t\t%f' % kl)
        print('')
        print('Total information gain:\t\t\t%f' % self.models[0].distribution.calculate_kl_divergence(
            self.models[-1].distribution))
        print('')
        print('########################################')
        print('')

    # Plot BMFMC distributions
    def plot_results(self, mc=False):

        if self.models[0].n_qoi == 1:

            # Determine bounds
            xmin = np.min([np.min(self.models[-1].model_evals_pred), np.min(self.models[0].model_evals_pred)])
            xmax = np.max([np.max(self.models[-1].model_evals_pred), np.max(self.models[0].model_evals_pred)])

            for i in range(self.n_models):
                # Plot
                color = 'C' + str(i)
                self.models[i].distribution.plot_kde(fignum=1, color=color, xmin=xmin, xmax=xmax,
                                                     title='BMFMC - approximate distributions')

            if mc and self.mc_model is not None:
                self.mc_model.distribution.plot_kde(fignum=1, color='k', linestyle='--',
                                                    xmin=xmin, xmax=xmax, title='BMFMC - approximate distributions')
            elif mc and self.mc_model is None:
                print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
                exit()

            plt.grid(b=True)
            plt.gcf().savefig('pngout/bmfmc_dists.png', dpi=300)

        else:
            # Plot marginals
            for k in range(self.models[-1].n_qoi):
                # Determine bounds
                xmin = np.min([np.min(self.models[-1].model_evals_pred[:, k]),
                               np.min(self.models[0].model_evals_pred[:, k])])
                xmax = np.max([np.max(self.models[-1].model_evals_pred[:, k]),
                               np.max(self.models[0].model_evals_pred[:, k])])

                for i in range(self.n_models):
                    # Plot
                    color = 'C' + str(i)
                    samples = self.models[i].distribution.samples[:, k]
                    samples = np.expand_dims(samples, axis=1)
                    marginal = Distribution(samples)
                    marginal.plot_kde(fignum=1, color=color, xmin=xmin, xmax=xmax)

                if mc and self.mc_model is not None:
                    samples = self.mc_model.distribution.samples[:, k]
                    samples = np.expand_dims(samples, axis=1)
                    marginal = Distribution(samples)
                    marginal.plot_kde(fignum=1, color='k', linestyle='--', xmin=xmin, xmax=xmax)

                elif mc and self.mc_model is None:
                    print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
                    exit()

                plt.grid(b=True)
                plt.gcf().savefig('pngout/bmfmc_dists_q%d.png' % (k + 1), dpi=300)
                plt.clf()

        if self.models[0].n_qoi == 2:

            sns.kdeplot(self.models[0].distribution.samples[:, 0], self.models[0].distribution.samples[:, 1],
                        shade=True, shade_lowest=False, cmap='Blues', label='Low-fidelity', color='C0')
            sns.kdeplot(self.models[-1].distribution.samples[:, 0], self.models[-1].distribution.samples[:, 1],
                        shade=True, shade_lowest=False, cmap='Reds', label='High-fidelity', color='C3')

            if mc and self.mc_model is not None:
                sns.kdeplot(self.mc_model.distribution.samples[:, 0], self.mc_model.distribution.samples[:, 1],
                            cmap='Greys', alpha=1.0, label='MC reference', color='Black')
            elif mc and self.mc_model is None:
                print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
                exit()
            plt.xlabel('$Q_1$')
            plt.ylabel('$Q_2$')
            plt.legend(loc='upper right')
            plt.title('BMFMC - approximate distributions')
            plt.grid(b=True)

            plt.gcf().savefig('pngout/bmfmc_dists.png', dpi=300)
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            plt.clf()

            self.models[0].distribution.plot_kde(title='BMFMC - low fidelity')
            plt.xlim([xmin, xmax])
            plt.ylim([ymin, ymax])
            plt.gcf().savefig('pngout/bmfmc_lf.png', dpi=300)
            plt.clf()

            self.models[-1].distribution.plot_kde(title='BMFMC - high fidelity')
            plt.xlim([xmin, xmax])
            plt.ylim([ymin, ymax])
            plt.gcf().savefig('pngout/bmfmc_hf.png', dpi=300)

            if mc and self.mc_model is not None:
                self.mc_model.distribution.plot_kde(title='Monte Carlo reference')
                plt.xlim([xmin, xmax])
                plt.ylim([ymin, ymax])
                plt.gcf().savefig('pngout/bmfmc_mc.png', dpi=300)
            elif mc and self.mc_model is None:
                print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
                exit()

        plt.clf()

    # Plot the regression models
    def plot_regression_models(self):

        for i in range(self.n_models - 1):
            lf_model = self.models[i]
            hf_model = self.models[i + 1]

            regression_model = self.regression_models[i]

            x_train = regression_model.x_train
            y_train = regression_model.y_train

            if hf_model.n_qoi == 1:

                x_pred = regression_model.x_pred
                mu = regression_model.mu
                sigma = regression_model.sigma

                # Sort to be able to use the plt.fill
                sort_indices = np.argsort(x_pred, axis=0)
                x_pred = np.squeeze(x_pred[sort_indices])
                y_pred = np.squeeze(mu[sort_indices])
                sigma = np.squeeze(sigma[sort_indices])

                utils.plot_1d_conf(x_pred, y_pred, sigma)
                utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k',
                                   label='Training', title='BMFMC - regression model', xlabel=lf_model.rv_name,
                                   ylabel=hf_model.rv_name)

            elif self.regression_type in ['decoupled_gaussian_process', 'decoupled_heteroscedastic_gaussian_process']:

                for k in range(hf_model.n_qoi):
                    x_pred = regression_model.x_pred[:, k]
                    mu = regression_model.mu[:, k]
                    sigma = regression_model.sigma[:, k]

                    # Sort to be able to use the plt.fill
                    sort_indices = np.argsort(x_pred, axis=0)
                    x_pred = np.squeeze(x_pred[sort_indices])
                    y_pred = np.squeeze(mu[sort_indices])
                    sigma = np.squeeze(sigma[sort_indices])

                    plt.grid(b=True)
                    utils.plot_1d_conf(x_pred, y_pred, sigma)
                    utils.plot_1d_data(x_train[:, k], y_train[:, k], marker='*', linestyle='', markersize=5, color='k',
                                       label='Training', title='BMFMC - regression model', xlabel=lf_model.rv_name,
                                       ylabel=hf_model.rv_name)

                    plt.gcf().savefig('pngout/bmfmc_regression_model_' + str(i + 1) + '_q' + str(k + 1) + '.png',
                                      dpi=300)
                    plt.clf()

            elif self.regression_type in ['shared_gaussian_process', 'shared_heteroscedastic_gaussian_process']:

                x_pred = regression_model.x_pred
                x_pred = x_pred.reshape(x_pred.size)
                mu = regression_model.mu
                mu = mu.reshape(mu.size)
                sigma = regression_model.sigma
                sigma = sigma.reshape(sigma.size)

                # Sort to be able to use the plt.fill
                sort_indices = np.argsort(x_pred, axis=0)
                x_pred = np.squeeze(x_pred[sort_indices])
                y_pred = np.squeeze(mu[sort_indices])
                sigma = np.squeeze(sigma[sort_indices])

                utils.plot_1d_conf(x_pred, y_pred, sigma)
                utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k',
                                   label='Training', title='BMFMC - regression model', xlabel=lf_model.rv_name,
                                   ylabel=hf_model.rv_name)

            elif hf_model.n_qoi == 2:

                sns.kdeplot(lf_model.distribution.samples[:, 0], lf_model.distribution.samples[:, 1],
                            shade=True, shade_lowest=False, cmap='Blues')
                sns.kdeplot(hf_model.distribution.samples[:, 0], hf_model.distribution.samples[:, 1],
                            shade=False, shade_lowest=False, cmap='Reds')
                colors = range(np.shape(x_train)[0])
                plt.scatter(x_train[:, 0], x_train[:, 1], c=colors, marker='o', s=30)
                plt.scatter(y_train[:, 0], y_train[:, 1], c=colors, marker='*', s=30)
                for k in range(np.shape(x_train)[0]):
                    plt.plot([x_train[k, 0], y_train[k, 0]], [x_train[k, 1], y_train[k, 1]], linestyle='--', color='k',
                             linewidth=1)
                plt.xlabel('$Q_1$')
                plt.ylabel('$Q_2$')
                plt.title('BMFMC - regression model')

            plt.grid(b=True)
            if self.n_models > 2:
                plt.gcf().savefig('pngout/bmfmc_regression_model_' + str(i + 1) + '.png', dpi=300)
            else:
                plt.gcf().savefig('pngout/bmfmc_regression_model.png', dpi=300)

            plt.clf()

    # Plot the approximate joint densities
    def plot_joint_densities(self):

        for i in range(self.n_models - 1):
            lf_model = self.models[i]
            hf_model = self.models[i + 1]

            samples = np.vstack([np.squeeze(lf_model.model_evals_pred), np.squeeze(hf_model.model_evals_pred)])
            utils.plot_2d_kde(samples=samples.T, title='Joint and marginals')

            plt.grid(b=True)
            if self.n_models > 2:
                plt.gcf().savefig('pngout/bmfmc_joint_dist_' + str(i + 1) + '.png', dpi=300)
            else:
                plt.gcf().savefig('pngout/bmfmc_joint_dist.png', dpi=300)

            plt.clf()
