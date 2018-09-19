import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

import utils
from Model import Model
from Regression import Regression


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
            self.mc_model.evaluate()

        self.mc_model.set_model_evals_pred(self.mc_model.model_evals)
        self.mc_model.create_distribution()

    # Calculate BMFMC estimator variance of the distribution mean
    def calculate_bmfmc_mean_estimator_variance(self):

        variance = 0.0
        for i in range(self.n_models-1):
            regression_model = self.regression_models[i]
            sigma = regression_model.sigma
            variance += np.mean(sigma ** 2, axis=0)

        return variance

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
        min = np.min(self.models[-1].distribution.samples)
        max = np.max(self.models[-1].distribution.samples)
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

        variance = 0.0
        min = np.percentile(self.models[-1].distribution.samples, 1)
        max = np.percentile(self.models[-1].distribution.samples, 99)
        y_range = np.linspace(min, max, n_vals)
        for k in range(self.n_models-1):
            # Get regression model
            regression_model = self.regression_models[k]
            mu = regression_model.mu
            sigma = regression_model.sigma

            # Approximate CDF
            n_lf = self.models[0].n_samples
            cdf_var = np.zeros((n_lf, y_range.shape[0]))

            print('')
            for i in range(y_range.shape[0]):
                print('CDF error bars (%d / %d), (%d / %d)' % (k+1, self.n_models-1, i + 1, y_range.shape[0]))
                for j in range(n_lf):
                    cdf_var[j, i] = stats.norm.cdf((y_range[i] - mu[j]) / (sigma[j] + 1e-15)) - stats.norm.cdf(
                        (y_range[i] - mu[j]) / (sigma[j] + 1e-15)) ** 2

            variance += np.mean(cdf_var, 0)

        return variance, y_range

    def calculate_bmfmc_expectation(self, fun=lambda x: x):

        samples = self.models[-1].model_evals_pred
        exp_samples = fun(samples)

        return np.mean(exp_samples, 0)

    def calculate_bmfmc_expectation_estimator_variance(self, fun=lambda x: x):

        variance = 0.0
        for i in range(self.n_models-1):
            # Get regression model
            regression_model = self.regression_models[i]
            mu = regression_model.mu
            sigma = regression_model.sigma

            # Approximate the variance of fun(QoI)
            n_lf = self.models[0].n_samples
            n_qoi = self.models[0].n_qoi
            exp_var_samples = np.zeros((n_lf, n_qoi))

            # Generate samples, get variance and average
            for j in range(n_lf):
                samples = np.random.randn(n_lf, n_qoi) * sigma[j] + mu[j]
                samples = fun(samples)
                exp_var_samples[j] = np.var(samples, axis=0)

            variance += np.mean(exp_var_samples, 0)

        return variance

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
        print('Mean estimator std:\t\t\t\t%s' % np.sqrt(self.calculate_bmfmc_mean_estimator_variance()))
        # mean = self.calculate_bmfmc_expectation(fun=lambda x: x)
        # print('Std estimator std:\t\t\t\t%s' % np.sqrt(self.calculate_bmfmc_expectation_estimator_variance(fun=lambda x: (x - mean)**2)))
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

            plt.gcf().savefig('pngout/bmfmc_dists.png', dpi=300)

        elif self.models[0].n_qoi == 2:

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

        else:
            print('BMFMC plotting only available for 1 and 2 QoIs.')
            exit()

        plt.clf()

    # Plot the regression models
    def plot_regression_models(self):

        for i in range(self.n_models - 1):
            lf_model = self.models[i]
            hf_model = self.models[i + 1]

            regression_model = self.regression_models[i]

            # This hack is necessary if using the decoupled GPs
            if isinstance(regression_model, list):
                x_train = np.zeros((np.shape(regression_model[0].x_train)[0], hf_model.n_qoi))
                y_train = np.zeros((np.shape(regression_model[0].x_train)[0], hf_model.n_qoi))
                for k in range(len(regression_model)):
                    x_train[:, k] = np.squeeze(regression_model[k].x_train)
                    y_train[:, k] = regression_model[k].y_train
                if hf_model.n_qoi == 1:
                    regression_model = regression_model[0]

            else:
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

            if self.n_models > 2:
                plt.gcf().savefig('pngout/bmfmc_joint_dist_' + str(i + 1) + '.png', dpi=300)
            else:
                plt.gcf().savefig('pngout/bmfmc_joint_dist.png', dpi=300)

            plt.clf()
