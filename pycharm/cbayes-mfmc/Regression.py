import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel
from sklearn.cluster import KMeans
from gp_extras.kernels import HeteroscedasticKernel
from scipy.stats import gaussian_kde as gkde
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import warnings


class Regression:

    # Class attributes:
    # - x_train: low-fidelity evaluations
    # - y_train: corresponding high-fidelity evaluations
    # - x_pred: low-fidelity points, where the high-fidelity solution is predicted
    # - regression_type: regression model type
    # - training_set_strategy:
    # - mu: predicted means
    # - sigma: predicted standard deviations
    # - regression_model: the scikit-learn model

    # Constructor
    def __init__(self, regression_type, training_set_strategy):

        self.x_train = None
        self.y_train = None
        self.x_pred = None
        self.regression_type = regression_type
        self.training_set_strategy = training_set_strategy
        self.mu = None
        self.sigma = None
        self.regression_model = None

    # Create the regression model and make high-fidelity predictions
    def build_regression_predict_and_sample(self):

        hf_model_evals_pred = None

        if self.regression_type == 'gaussian_process':

            self.regression_model = []
            n_qoi = self.x_train.shape[1]
            self.mu = np.zeros(self.x_pred.shape)
            self.sigma = np.zeros(self.x_pred.shape)
            hf_model_evals_pred = np.zeros(self.x_pred.shape)

            for i in range(n_qoi):

                # Fit a GP regression model to approximate p(q_l|q_l-1)
                kernel = ConstantKernel() + ConstantKernel() * RBF(np.ones(n_qoi)) + WhiteKernel()
                self.regression_model.append(gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                                                                       n_restarts_optimizer=0))
                self.regression_model[i].fit(self.x_train, self.y_train[:, i])

                # Predict q_l|q_l-1 at all low-fidelity samples
                self.mu[:, i], self.sigma[:, i] = self.regression_model[i].predict(self.x_pred, return_std=True)

                # Generate high-fidelity samples from the predictions
                for j in range(self.mu.shape[0]):
                    hf_model_evals_pred[j, i] = self.mu[j, i] + self.sigma[j, i] * np.random.randn()

        elif self.regression_type == 'consistent_gaussian_process':

            self.regression_model = []
            n_qoi = self.x_train.shape[1]
            self.mu = np.zeros(self.x_pred.shape)
            self.sigma = np.zeros(self.x_pred.shape)
            hf_model_evals_pred = np.zeros(self.x_pred.shape)

            for i in range(n_qoi):

                # Enrich x_train
                if i > 0:
                    x_train = np.hstack([self.x_train, self.y_train[:, 0:i].reshape((-1, i))])
                else:
                    x_train = self.x_train

                # Fit a GP regression model to approximate p(q_l|q_l-1)
                kernel = ConstantKernel() + ConstantKernel() * RBF(np.ones(n_qoi+i)) + WhiteKernel()
                self.regression_model.append(gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                                                                       n_restarts_optimizer=0))
                self.regression_model[i].fit(x_train, self.y_train[:, i])

                # Enrich x_pred
                if i > 0:
                    x_pred = np.hstack([self.x_pred, hf_model_evals_pred[:, 0:i].reshape((-1, i))])
                    # x_pred = np.hstack([self.x_pred, self.mu[:, 0:i].reshape((-1, i))])
                else:
                    x_pred = self.x_pred

                # Predict q_l|q_l-1 at all low-fidelity samples
                self.mu[:, i], self.sigma[:, i] = self.regression_model[i].predict(x_pred, return_std=True)

                # Generate high-fidelity samples from the predictions
                for j in range(self.mu.shape[0]):
                    hf_model_evals_pred[j, i] = self.mu[j, i] + self.sigma[j, i] * np.random.randn()

        elif self.regression_type == 'gaussian_process_kde':

            self.regression_model = []
            n_qoi = self.x_train.shape[1]
            self.mu = np.zeros(self.x_pred.shape)
            self.sigma = np.zeros(self.x_pred.shape)
            hf_model_evals_pred = np.zeros(self.x_pred.shape)

            for i in range(n_qoi):

                # Fit a GP regression model to approximate p(q_l|q_l-1)
                kernel = ConstantKernel() + RBF(np.ones(n_qoi)) + WhiteKernel()
                self.regression_model.append(gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                                                                       n_restarts_optimizer=0))
                self.regression_model[i].fit(self.x_train, self.y_train[:, i])
                                
                # Predict q_l|q_l-1 at all low-fidelity samples
                self.mu[:, i], self.sigma[:, i] = self.regression_model[i].predict(self.x_pred, return_std=True)

                mu_train = self.regression_model[i].predict(self.x_train, return_std=False)
                noise_train = self.y_train[:, i] - mu_train
                nmin = np.min(noise_train)
                nmax = np.max(noise_train)
                
                # plt.figure(1)
                # samples = np.vstack([np.squeeze(self.x_train), np.squeeze(noise_train)])
                # df = pd.DataFrame(samples.T, columns=['$Q_1$', 'Noise'])
                # g = sns.jointplot(x='$Q_1$', y='Noise', data=df, kind='kde', color='C0', shade=True,
                #                   shade_lowest=True, cmap='Blues')
                # g.plot_joint(plt.scatter, c='k', alpha=0.3, s=20, linewidth=0.0, marker='o')
                # g.ax_joint.collections[0].set_alpha(0)
                # # g.ax_joint.legend_.remove()
                # g.set_axis_labels('$Q_1$', 'Noise')
                # plt.subplots_adjust(top=0.95)
                # plt.gcf().subplots_adjust(left=0.15)
                # # plt.show()
                # plt.gcf().savefig('output/mfmc_noise_model.pdf', dpi=300)
                
                joint_kde = gkde([self.x_train[:, i], noise_train])
                
                # Generate high-fidelity samples from the predictions
                for j in range(self.mu.shape[0]):
                    hf_model_evals_pred[j, i] = self.mu[j, i]
                                        
                    # given x_pred, generate samples of y and average to get normalizing factor for slice of kde
                    Nsamp = 100
                    nvals = np.random.uniform(low=nmin, high=nmax, size=Nsamp)
                    kde_slice_samp = joint_kde([self.x_pred[j, i] * np.ones(Nsamp), nvals])
                    normfactor = np.mean(kde_slice_samp)
                    kde_slice_samp *= 1.0 / normfactor
                    ratio = np.divide(kde_slice_samp, 1.0 / (nmax - nmin) * np.ones(Nsamp))
                    ratio *= 1.0 / np.max(ratio)
                    foundsamp = False
                    ii = 0
                    while not foundsamp and ii < Nsamp:
                        if ratio[ii] > np.random.uniform(low=0, high=1, size=1):
                            foundsamp = True
                            hf_model_evals_pred[j, i] += nvals[ii]
                        else:
                            ii += 1
                    if not foundsamp:
                        print('Rejection sampling failed. Using the mean prediction without noise.')

        elif self.regression_type == 'decoupled_gaussian_process':

            self.regression_model = []
            self.mu = np.zeros(self.x_pred.shape)
            self.sigma = np.zeros(self.x_pred.shape)
            hf_model_evals_pred = np.zeros(self.x_pred.shape)

            for i in range(self.x_train.shape[1]):

                # Fit a GP regression model to approximate p(q_l|q_l-1)
                kernel = ConstantKernel() + ConstantKernel() * RBF() + WhiteKernel()
                self.regression_model.append(gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-6))
                self.regression_model[i].fit(np.expand_dims(self.x_train[:, i], axis=1), self.y_train[:, i])

                # Predict q_l|q_l-1 at all low-fidelity samples
                self.mu[:, i], self.sigma[:, i] = self.regression_model[i].predict(
                    np.expand_dims(self.x_pred[:, i], axis=1), return_std=True)

                # Generate high-fidelity samples from the predictions
                for j in range(self.mu.shape[0]):
                    hf_model_evals_pred[j, i] = self.mu[j, i] + self.sigma[j, i] * np.random.randn()

        elif self.regression_type == 'heteroscedastic_gaussian_process':

            self.regression_model = []
            n_qoi = self.x_train.shape[1]
            self.mu = np.zeros(self.x_pred.shape)
            self.sigma = np.zeros(self.x_pred.shape)
            hf_model_evals_pred = np.zeros(self.x_pred.shape)

            for i in range(n_qoi):

                # Fit a heteroscedastic GP regression model with spatially varying noise to approximate p(q_l|q_l-1)
                # See here for more info: https://github.com/jmetzen/gp_extras/
                prototypes = KMeans(n_clusters=5).fit(self.x_train).cluster_centers_
                kernel = ConstantKernel() + ConstantKernel() * RBF(np.ones(self.y_train.shape[1])) \
                    + HeteroscedasticKernel.construct(prototypes, 1e-3, (1e-10, 5e1), gamma=5.0, gamma_bounds="fixed")
                self.regression_model.append(gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-6))
                self.regression_model[i].fit(self.x_train, self.y_train[:, i])

                # Predict q_l|q_l-1 at all low-fidelity samples
                self.mu[:, i], self.sigma[:, i] = self.regression_model[i].predict(self.x_pred, return_std=True)

                # Generate high-fidelity samples from the predictions
                for j in range(self.mu.shape[0]):
                    hf_model_evals_pred[j, i] = self.mu[j, i] + self.sigma[j, i] * np.random.randn()

        elif self.regression_type == 'consistent_heteroscedastic_gaussian_process':

            self.regression_model = []
            n_qoi = self.x_train.shape[1]
            self.mu = np.zeros(self.x_pred.shape)
            self.sigma = np.zeros(self.x_pred.shape)
            hf_model_evals_pred = np.zeros(self.x_pred.shape)

            for i in range(n_qoi):

                # Enrich x_train
                if i > 0:
                    x_train = np.hstack([self.x_train, self.y_train[:, 0:i].reshape((-1, i))])
                else:
                    x_train = self.x_train

                # Fit a heteroscedastic GP regression model with spatially varying noise to approximate p(q_l|q_l-1)
                # See here for more info: https://github.com/jmetzen/gp_extras/
                prototypes = KMeans(n_clusters=5).fit(x_train).cluster_centers_
                kernel = ConstantKernel() + ConstantKernel() * RBF(np.ones(n_qoi+i)) \
                    + HeteroscedasticKernel.construct(prototypes, 1e-3, (1e-10, 5e1), gamma=5.0, gamma_bounds="fixed")
                self.regression_model.append(gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-6))
                self.regression_model[i].fit(x_train, self.y_train[:, i])

                # Enrich x_pred
                if i > 0:
                    x_pred = np.hstack([self.x_pred, hf_model_evals_pred[:, 0:i].reshape((-1, i))])
                    # x_pred = np.hstack([self.x_pred, self.mu[:, 0:i].reshape((-1, i))])
                else:
                    x_pred = self.x_pred

                # Predict q_l|q_l-1 at all low-fidelity samples
                self.mu[:, i], self.sigma[:, i] = self.regression_model[i].predict(x_pred, return_std=True)

                # Generate high-fidelity samples from the predictions
                for j in range(self.mu.shape[0]):
                    hf_model_evals_pred[j, i] = self.mu[j, i] + self.sigma[j, i] * np.random.randn()

        elif self.regression_type == 'decoupled_heteroscedastic_gaussian_process':

            self.regression_model = []
            self.mu = np.zeros(self.x_pred.shape)
            self.sigma = np.zeros(self.x_pred.shape)
            hf_model_evals_pred = np.zeros(self.x_pred.shape)

            for i in range(self.x_train.shape[1]):

                # Fit a heteroscedastic GP regression model with spatially varying noise to approximate p(q_l|q_l-1)
                # See here for more info: https://github.com/jmetzen/gp_extras/
                prototypes = KMeans(n_clusters=5).fit(np.expand_dims(self.x_pred[:, i], axis=1)).cluster_centers_
                kernel = ConstantKernel() + ConstantKernel() * RBF() \
                         + HeteroscedasticKernel.construct(prototypes, 1e-3, (1e-10, 5e1), gamma=5.0,
                                                           gamma_bounds="fixed")
                self.regression_model.append(gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-6))
                self.regression_model[i].fit(np.expand_dims(self.x_train[:, i], axis=1), self.y_train[:, i])

                # Predict q_l|q_l-1 at all low-fidelity samples
                self.mu[:, i], self.sigma[:, i] = self.regression_model[i].predict(
                    np.expand_dims(self.x_pred[:, i], axis=1), return_std=True)

                # Generate high-fidelity samples from the predictions
                for j in range(self.mu.shape[0]):
                    hf_model_evals_pred[j, i] = self.mu[j, i] + self.sigma[j, i] * np.random.randn()

        else:
            print('Unknown regression model %s.' % self.regression_type)
            exit()

        # Replace INF or NAN predictions
        if np.isinf(hf_model_evals_pred).any() or np.isnan(hf_model_evals_pred).any():
            warnings.warn('Detected INF or NAN values in the predictions, replacing them with the sample mean.')
            mask = np.isnan(hf_model_evals_pred) | np.isinf(hf_model_evals_pred)
            hf_model_evals_pred[mask] = np.mean(hf_model_evals_pred[~mask], axis=0)

        return hf_model_evals_pred

    # Create the training set for the regression
    def create_training_set(self, lf_model, hf_model, id, regression_models):

        x_train = None

        if self.training_set_strategy is 'support_covering':

            x_train_linspace = None

            # Create a uniform grid across the support of p(q)
            if hf_model.n_qoi is 1:
                sup_min = np.min(lf_model.model_evals_pred[:, 0])
                sup_max = np.max(lf_model.model_evals_pred[:, 0])
                x_train_linspace = np.linspace(sup_min, sup_max, num=hf_model.n_evals)
                x_train_linspace = np.reshape(x_train_linspace, (hf_model.n_evals, hf_model.n_qoi))

            elif hf_model.n_qoi is 2:
                ab_num = int(np.sqrt(hf_model.n_evals))
                hf_model.n_evals = ab_num ** 2
                sup_min = np.min(lf_model.model_evals_pred[:, 0])
                sup_max = np.max(lf_model.model_evals_pred[:, 0])
                a = np.linspace(sup_min, sup_max, num=ab_num)
                sup_min = np.min(lf_model.model_evals_pred[:, 1])
                sup_max = np.max(lf_model.model_evals_pred[:, 1])
                b = np.linspace(sup_min, sup_max, num=ab_num)
                aa, bb = np.meshgrid(a, b)
                x_train_linspace = np.reshape(np.vstack([aa, bb]), (2, hf_model.n_evals)).T

            elif hf_model.n_qoi is 3:
                abc_num = int(np.power(hf_model.n_evals, 1. / 3))
                hf_model.n_evals = abc_num ** 3
                sup_min = np.min(lf_model.model_evals_pred[:, 0])
                sup_max = np.max(lf_model.model_evals_pred[:, 0])
                a = np.linspace(sup_min, sup_max, num=abc_num)
                sup_min = np.min(lf_model.model_evals_pred[:, 1])
                sup_max = np.max(lf_model.model_evals_pred[:, 1])
                b = np.linspace(sup_min, sup_max, num=abc_num)
                sup_min = np.min(lf_model.model_evals_pred[:, 2])
                sup_max = np.max(lf_model.model_evals_pred[:, 2])
                c = np.linspace(sup_min, sup_max, num=abc_num)
                aa, bb, cc = np.meshgrid(a, b, c)
                x_train_linspace = np.reshape(np.vstack([aa, bb, cc]), (3, hf_model.n_evals)).T

            else:
                print('Support covering strategy only available for up to 3 QoIs. Use sampling.')
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
                else:
                    regression_model = regression_models[id - 1]
                    mu = regression_model.mu

                    idx = (np.linalg.norm(mu - x_train_linspace[i, :], axis=1, ord=1)).argmin()

                    # There are 3 ways to choose x_train. It seems to pay off to invest in additional evaluations in 1),
                    # using 2) and 3) will most likely increase the variance in the correlations:

                    # 1) Evaluating the low-fidelity
                    x_train[i, :] = lf_model.eval_fun(lf_model.rv_samples_pred[idx, :])

                    # 2) Using the predicted mean
                    # x_train[i, :] = mu[idx, :]

                    # 3) Using the predicted sample
                    # x_train[i, :] = lf_model.model_evals_pred[idx, :]

                hf_rv_samples[i, :] = lf_model.rv_samples_pred[idx, :]

            # Assign model evaluation points to the high-fidelity model
            hf_model.set_rv_samples(hf_rv_samples)

        elif self.training_set_strategy is 'sampling':

            # Get some random variable samples
            indices = np.random.choice(range(lf_model.rv_samples_pred.shape[0]), size=hf_model.n_evals, replace=False)

            # Get the corresponding lower-fidelity evaluations
            # Those are only available for the lowest-fidelity model and need to be computed for any other model
            if id == 0:
                x_train = lf_model.model_evals_pred[indices, :]

            else:
                # There are 3 ways to choose x_train. It seems to pay off to invest in additional evaluations in 1),
                # using 2) and 3) will most likely increase the variance in the correlations:

                # 1) Evaluating the low-fidelity
                x_train = np.zeros((hf_model.n_evals, lf_model.n_qoi))
                for i in range(hf_model.n_evals):
                    x_train[i, :] = lf_model.eval_fun(lf_model.rv_samples_pred[indices[i], :])

                # 2) Using the predicted mean
                # regression_model = regression_models[id - 1]
                # mu = regression_model.mu
                # x_train = mu[indices, :]

                # 3) Using the predicted sample
                # x_train = lf_model.model_evals_pred[indices]

            # Assign model evaluation points to the high-fidelity model
            hf_rv_samples = lf_model.rv_samples_pred[indices, :]
            hf_model.set_rv_samples(hf_rv_samples)

        elif self.training_set_strategy is 'fixed':

            # We just choose the first low-fidelity samples
            indices = range(hf_model.n_evals)

            # Get the corresponding lower-fidelity evaluations
            x_train = lf_model.model_evals_pred[indices]

            # Assign model evaluation points to the high-fidelity model
            hf_rv_samples = lf_model.rv_samples_pred[indices]
            hf_model.set_rv_samples(hf_rv_samples)

        else:
            print('Unknown training set selection strategy.')
            exit()

        return x_train

    def predict_and_sample(self, x_pred):

        if self.regression_model is None:
            print('No regression model available. Make sure you have called build_regression_predict_and_sample().')
            exit()

        mu = np.zeros(x_pred.shape)
        sigma = np.zeros(x_pred.shape)
        hf_model_evals_pred = np.zeros(x_pred.shape)

        if self.regression_type in ['gaussian_process', 'heteroscedastic_gaussian_process']:

            for i in range(x_pred.shape[1]):

                # Predict q_l|q_l-1 at all low-fidelity samples
                mu[:, i], sigma[:, i] = self.regression_model[i].predict(x_pred, return_std=True)

                # Generate high-fidelity samples from the predictions
                for j in range(mu.shape[0]):
                    hf_model_evals_pred[j, i] = mu[j, i] + sigma[j, i] * np.random.randn()

        elif self.regression_type in ['consistent_gaussian_process', 'consistent_heteroscedastic_gaussian_process']:

            for i in range(x_pred.shape[1]):

                # Enrich x_pred
                if i > 0:
                    x_pred = np.hstack([x_pred, hf_model_evals_pred[:, 0:i].reshape((-1, i))])

                # Predict q_l|q_l-1 at all low-fidelity samples
                mu[:, i], sigma[:, i] = self.regression_model[i].predict(x_pred, return_std=True)

                # Generate high-fidelity samples from the predictions
                for j in range(mu.shape[0]):
                    hf_model_evals_pred[j, i] = mu[j, i] + sigma[j, i] * np.random.randn()

        elif self.regression_type in ['decoupled_gaussian_process', 'decoupled_heteroscedastic_gaussian_process']:

            for i in range(x_pred.shape[1]):

                # Predict q_l|q_l-1 at all low-fidelity samples
                mu[:, i], sigma[:, i] = self.regression_model[i].predict(np.expand_dims(x_pred[:, i], axis=1),
                                                                         return_std=True)

                # Generate high-fidelity samples from the predictions
                for j in range(mu.shape[0]):
                    hf_model_evals_pred[j, i] = mu[j, i] + sigma[j, i] * np.random.randn()

        elif self.regression_type == 'gaussian_process_kde':

            for i in range(x_pred.shape[1]):

                # Predict q_l|q_l-1 at all low-fidelity samples
                mu[:, i], sigma[:, i] = self.regression_model[i].predict(x_pred, return_std=True)

                mu_train, sigma_train = self.regression_model[i].predict(self.x_train, return_std=True)
                noise_train = self.y_train[:, i] - mu_train
                joint_kde = gkde([self.x_train[:, i], noise_train])
                nmin = np.min(noise_train)
                nmax = np.max(noise_train)
                
                # Generate high-fidelity samples from the predictions
                for j in range(mu.shape[0]):
                    hf_model_evals_pred[j, i] = mu[j, i]
                                        
                    # given x_pred, generate samples of y and average to get normalizing factor for slice of kde
                    Nsamp = 100
                    nvals = np.random.uniform(low=nmin, high=nmax, size=Nsamp)
                    kde_slice_samp = joint_kde([self.x_pred[j, i] * np.ones(Nsamp), nvals])
                    normfactor = np.mean(kde_slice_samp)
                    kde_slice_samp *= 1.0 / normfactor
                    ratio = np.divide(kde_slice_samp, 1.0 / (nmax - nmin) * np.ones(Nsamp))
                    ratio *= 1.0 / np.max(ratio)
                    foundsamp = False
                    ii = 0
                    while not foundsamp:
                        if ratio[ii] > np.random.uniform(low=0, high=1, size=1):
                            foundsamp = True
                            hf_model_evals_pred[j, i] += nvals[ii]
                        else:
                            ii += 1

        else:
            print('Unknown regression model %s.' % self.regression_type)
            exit()

        return hf_model_evals_pred
