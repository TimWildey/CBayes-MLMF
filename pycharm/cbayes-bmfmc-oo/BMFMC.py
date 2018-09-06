import matplotlib.pyplot as plt
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Product, DotProduct
from sklearn.cluster import KMeans
from gp_extras.kernels import HeteroscedasticKernel
import seaborn as sns

import utils
from Model import Model


class BMFMC:

    # Class attributes:
    # - models: a list of all models for BMFMC (i.e. at least one high-fidelity and one low-fidelity model)
    # - mc_model: the Monte Carlo reference model (computed only if calculate_mc_reference() is called)
    # - n_models: number of models
    # - training_set_selection: strategy for training set selection
    # - regression_model: regression model for the conditionals p(q_l|q_l-1)

    # Constructor
    def __init__(self, models, training_set_strategy, regression_type):

        self.models = models
        self.mc_model = None
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

        x_train = []

        if self.training_set_strategy == 'support_covering' or self.training_set_strategy == 'support_covering_adaptive':

            if len(self.regression_models) == id:
                # Create a uniform grid across the support of p(q)

                if hf_model.n_qoi is 1:
                    x_train_linspace = np.linspace(lf_model.model_evals_pred[:, 0].min(),
                                                   lf_model.model_evals_pred[:, 0].max(),
                                                   num=hf_model.n_evals)
                    x_train_linspace = np.reshape(x_train_linspace, (hf_model.n_evals, hf_model.n_qoi))

                elif hf_model.n_qoi is 2:
                    ab_num = int(np.sqrt(hf_model.n_evals))
                    hf_model.n_evals = ab_num ** 2
                    a = np.linspace(lf_model.model_evals_pred[:, 0].min(), lf_model.model_evals_pred[:, 0].max(),
                                    num=ab_num)
                    b = np.linspace(lf_model.model_evals_pred[:, 1].min(), lf_model.model_evals_pred[:, 1].max(),
                                    num=ab_num)
                    aa, bb = np.meshgrid(a, b)
                    x_train_linspace = np.reshape(np.vstack([aa, bb]), (2, hf_model.n_evals)).T

                else:
                    print('Support covering strategy only available for 1 or 2 QoIs.')
                    exit()

            else:
                # Create new points between existing ones
                if hf_model.n_qoi is 1:
                    sorted_xtrain = np.sort(self.regression_models[id].X_train_, axis=0)
                    diffs = np.diff(sorted_xtrain, n=1, axis=0)
                    diffs = np.reshape(diffs, (np.shape(diffs)[0], hf_model.n_qoi))
                    x_train_linspace = sorted_xtrain[:-1, :] + 0.5 * diffs

                else:
                    print('Support covering adaptive strategy only available for 1 QoI.')
                    exit()

            n_train = np.shape(x_train_linspace)[0]

            # Find the lower-fidelity samples closest to the grid points and get the corresponding lambdas
            x_train = np.zeros((n_train, hf_model.n_qoi))
            hf_rv_samples = np.zeros((n_train, lf_model.n_random))
            for i in range(n_train):

                # This is only exact for the lowest fidelity regression model
                if id == 0:
                    idx = (np.linalg.norm(lf_model.model_evals_pred - x_train_linspace[i, :], axis=1, ord=1)).argmin()
                    x_train[i, :] = lf_model.model_evals_pred[idx, :]

                # For any other regression model, finding the correct x,y pair is a noisy task.
                # We use the mean predictions of the GP to do that.
                else:
                    regression_model = self.regression_models[id - 1]
                    mu = regression_model.predict(self.models[id - 1].model_evals_pred, return_std=False)
                    idx = (np.linalg.norm(mu - x_train_linspace[i, :], axis=1, ord=1)).argmin()
                    x_train[i, :] = lf_model.eval_fun(lf_model.rv_samples_pred[idx, :])

                hf_rv_samples[i, :] = lf_model.rv_samples_pred[idx, :]

            # Assign model evaluation points to the high-fidelity model
            hf_model.set_rv_samples(hf_rv_samples)

            if self.training_set_strategy == 'support_covering':
                # No adaptivity
                self.adaptive = False

        elif self.training_set_strategy == 'sampling' or self.training_set_strategy == 'sampling_adaptive':

            # Get some random variable samples
            indices = np.random.choice(range(lf_model.rv_samples_pred.shape[0]), size=hf_model.n_evals, replace=False)

            # Get the corresponding lower-fidelity evaluations
            # Those are only available for the lowest-fidelity model and need to be computed for any other model
            if id > 0:
                x_train = np.zeros((hf_model.n_evals, lf_model.n_qoi))
                for i in range(hf_model.n_evals):
                    x_train[i, :] = lf_model.eval_fun(lf_model.rv_samples_pred[indices[i], :])
            elif id == 0:
                x_train = lf_model.model_evals_pred[indices, :]
            else:
                print('Invalid model id. Something went very wrong.')
                exit()

            # Assign model evaluation points to the high-fidelity model
            hf_rv_samples = lf_model.rv_samples_pred[indices, :]
            hf_model.set_rv_samples(hf_rv_samples)

            if self.training_set_strategy == 'sampling':
                # No adaptivity
                self.adaptive = False

        else:
            print('Unknown training set selection strategy.')
            exit()

        return x_train

    # Create the regression model and make high-fidelity predictions
    def build_regression_predict_and_sample(self, x_train, y_train, x_pred, id):

        hf_model_evals_pred = None
        regression_model = None

        if self.regression_type == 'gaussian_process':

            # Fit a GP regression model to approximate p(q_l|q_l-1)
            kernel = ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() + DotProduct()
            # kernel = Matern() + WhiteKernel()
            regression_model = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
            regression_model.fit(x_train, y_train)

            # Predict q_l|q_l-1 at all low-fidelity samples
            mu, sigma = regression_model.predict(x_pred, return_std=True)

            # Generate high-fidelity samples from the predictions
            hf_model_evals_pred = np.zeros((mu.shape[0], mu.shape[1]))
            for i in range(mu.shape[0]):
                hf_model_evals_pred[i, :] = mu[i, :] + sigma[i] * np.random.randn(1, mu.shape[1])

        elif self.regression_type == 'decoupled_gaussian_processes':

            # Fit a GP regression model to approximate p(q_l|q_l-1)
            kernel = ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() + DotProduct()

            hf_model_evals_pred = np.zeros((x_pred.shape[0], x_pred.shape[1]))
            regression_model = []
            for k in range(x_train.shape[1]):

                regression_model.append(gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-10))
                regression_model[k].fit(np.expand_dims(x_train[:, k], axis=1), y_train[:, k])

                # Predict q_l|q_l-1 at all low-fidelity samples
                mu, sigma = regression_model[k].predict(np.expand_dims(x_pred[:, k], axis=1), return_std=True)

                # Generate high-fidelity samples from the predictions
                for i in range(mu.shape[0]):
                    hf_model_evals_pred[i, k] = mu[i] + sigma[i] * np.random.randn()

        elif self.regression_type == 'heteroscedastic_gaussian_process':

            # Fit a heteroscedastic GP regression model with spatially varying noise to approximate p(q_l|q_l-1)
            # See here for more info: https://github.com/jmetzen/gp_extras/
            prototypes = KMeans(n_clusters=10).fit(x_train).cluster_centers_
            kernel = ConstantKernel(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0)) + HeteroscedasticKernel.construct(
                prototypes, 1e-3, (1e-10, 50.0), gamma=5.0, gamma_bounds="fixed")
            regression_model = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
            regression_model.fit(x_train, y_train)

            # Predict q_l|q_l-1 at all low-fidelity samples
            mu, sigma = regression_model.predict(x_pred, return_std=True)

            # Generate high-fidelity samples from the predictions
            hf_model_evals_pred = np.zeros((mu.shape[0], mu.shape[1]))
            for i in range(mu.shape[0]):
                hf_model_evals_pred[i, :] = mu[i, :] + sigma[i] * np.random.randn(1, mu.shape[1])

        else:
            print('Unknown regression model %s.' % self.regression_type)
            exit()

        # Save regression model
        if len(self.regression_models) == id:
            self.regression_models.append(regression_model)
        elif len(self.regression_models) == id + 1:
            self.regression_models[id] = regression_model
        else:
            print('This is not supposed to happen. Something went very wrong.')
            exit()

        return hf_model_evals_pred

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

        self.mc_model = Model(eval_fun=self.models[-1].eval_fun, n_evals=self.models[0].n_evals,
                              n_qoi=self.models[-1].n_qoi, rv_samples=self.models[0].rv_samples,
                              rv_samples_pred=self.models[0].rv_samples, label='MC reference',
                              rv_name=self.models[-1].rv_name)
        self.mc_model.evaluate()
        self.mc_model.set_model_evals_pred(self.mc_model.model_evals)
        self.mc_model.create_distribution()

    # Calculate BMFMC estimator variance of the distribution mean
    def calculate_bmfmc_mean_estimator_variance(self):

        if self.n_models > 1 or self.models[0].n_qoi > 1:
            print('This only works for one low-fidelity model and one QoI.')
            exit()

        regression_model = self.regression_models[-1]
        x_pred = self.models[0].model_evals_pred
        _, sigma = regression_model.predict(x_pred, return_std=True)

        return np.sqrt(np.mean(sigma ** 2, axis=0))

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
            plt.clf()

        elif self.models[0].n_qoi == 2:

            sns.kdeplot(self.models[0].distribution.samples[:, 0], self.models[0].distribution.samples[:, 1],
                        shade=True, shade_lowest=False, cmap='Blues')
            sns.kdeplot(self.models[-1].distribution.samples[:, 0], self.models[-1].distribution.samples[:, 1],
                        shade=True, shade_lowest=False, cmap='Reds')

            if mc and self.mc_model is not None:
                sns.kdeplot(self.mc_model.distribution.samples[:, 0], self.mc_model.distribution.samples[:, 1],
                            cmap='Greys', alpha=1.0)
            elif mc and self.mc_model is None:
                print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
                exit()

            plt.gcf().savefig('pngout/bmfmc_dists.png', dpi=300)
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            plt.clf()

            self.models[0].distribution.plot_kde()
            plt.xlim([xmin, xmax])
            plt.ylim([ymin, ymax])
            plt.gcf().savefig('pngout/bmfmc_lf.png', dpi=300)
            plt.clf()

            self.models[-1].distribution.plot_kde()
            plt.xlim([xmin, xmax])
            plt.ylim([ymin, ymax])
            plt.gcf().savefig('pngout/bmfmc_hf.png', dpi=300)
            plt.clf()

            if mc and self.mc_model is not None:
                self.mc_model.distribution.plot_kde()
                plt.xlim([xmin, xmax])
                plt.ylim([ymin, ymax])
                plt.gcf().savefig('pngout/bmfmc_mc.png', dpi=300)
                plt.clf()
            elif mc and self.mc_model is None:
                print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
                exit()

        else:
            print('BMFMC plotting only available for 1 and 2 QoIs.')
            exit()

    # Plot the regression models
    def plot_regression_models(self):

        for i in range(self.n_models - 1):
            lf_model = self.models[i]
            hf_model = self.models[i + 1]

            regression_model = self.regression_models[i]

            # This hack is necessary if using the decoupled_gaussian_process
            if isinstance(regression_model, list):
                x_train = np.zeros((np.shape(regression_model[0].X_train_)[0], hf_model.n_qoi))
                y_train = np.zeros((np.shape(regression_model[0].X_train_)[0], hf_model.n_qoi))
                for k in range(len(regression_model)):
                    x_train[:, k] = np.squeeze(regression_model[k].X_train_)
                    y_train[:, k] = regression_model[k].y_train_
                if hf_model.n_qoi == 1:
                    regression_model = regression_model[0]

            else:
                x_train = regression_model.X_train_
                y_train = regression_model.y_train_

            if hf_model.n_qoi == 1:

                x_pred = lf_model.model_evals_pred
                mu, sigma = regression_model.predict(x_pred, return_std=True)

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
            utils.plot_2d_kde(samples=samples.T,
                              xlabel=lf_model.rv_name, ylabel=hf_model.rv_name)

            if self.n_models > 2:
                plt.gcf().savefig('pngout/bmfmc_joint_dist_' + str(i + 1) + '.png', dpi=300)
            else:
                plt.gcf().savefig('pngout/bmfmc_joint_dist.png', dpi=300)

            plt.clf()
