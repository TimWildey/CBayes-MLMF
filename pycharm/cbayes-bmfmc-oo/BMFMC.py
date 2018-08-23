import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Product, DotProduct
import matplotlib.pyplot as plt
from Model import Model
import utils


class BMFMC:

    # Class attributes:
    # - models: a list of all models for BMFMC (i.e. at least one high-fidelity and one low-fidelity model)
    # - mc_model: the Monte Carlo reference model (computed only if calculate_mc_reference() is called)
    # - n_models: number of models
    # - training_set_selection: strategy for training set selection
    # - regression_model: regression model for the conditionals p(q_l|q_l-1)
    # - fignum: figure counter for plotting

    def __init__(self, models, training_set_strategy, regression_type):

        self.models = models
        self.mc_model = 0
        self.n_models = len(models)
        self.training_set_strategy = training_set_strategy
        self.regression_type = regression_type
        self.regression_models = []
        self.fignum = 10

        if len(models) > 2:
            print('More than one low-fidelity model not tested yet.')
            exit()

    def apply_bmfmc_framework(self):

        for i in range(self.n_models-1):

            lf_model = self.models[i]
            hf_model = self.models[i+1]

            # 1) Evaluate the lowest-fidelity model
            if i == 0:
                lf_model.evaluate()
                lf_model.set_model_evals_pred(lf_model.model_evals)
                lf_model.create_distribution()

            # 2) Select lower-fidelity model evaluation points and evaluate the next higher-fidelity model
            x_train = self.create_training_set(lf_model, hf_model)
            y_train = hf_model.evaluate()

            # 3) Fit a regression model to approximate p(q_l|q_l-1), predict q_l|q_l-1 at all low-fidelity samples,
            #    generate low-fidelity samples from the predictions and create a distribution
            hf_model_evals_pred = self.build_regression_predict_and_sample(x_train=x_train, y_train=y_train,
                                                     x_pred=lf_model.model_evals_pred)
            hf_model.set_rv_samples_pred(lf_model.rv_samples_pred)
            hf_model.set_model_evals_pred(hf_model_evals_pred.reshape((self.models[0].n_evals, hf_model.n_qoi)))
            hf_model.create_distribution()

        return self.models[-1].model_evals_pred

    def create_training_set(self, lf_model, hf_model):

        x_train = []
        if self.training_set_strategy == 'support_covering':

            # Create a uniform grid across the support of p(q)
            x_train_linspace = np.linspace(lf_model.model_evals_pred.min(), lf_model.model_evals_pred.max(),
                                           num=hf_model.n_evals)

            # Find the lower-fidelity samples closest to the grid points and get the corresponding lambdas
            x_train = np.zeros((hf_model.n_evals, hf_model.n_qoi))
            hf_rv_samples = np.zeros((hf_model.n_evals, lf_model.n_random))
            for i in range(hf_model.n_evals):
                idx = (np.abs(lf_model.model_evals_pred - x_train_linspace[i])).argmin()
                x_train[i] = lf_model.model_evals_pred[idx]
                hf_rv_samples[i, :] = lf_model.rv_samples_pred[idx, :]

            hf_model.set_rv_samples(hf_rv_samples)

        elif self.training_set_strategy == 'samples':

            print('Not implemented yet.')
            exit()

        else:
            print('Unknown training set selection strategy.')
            exit()

        return x_train

    def build_regression_predict_and_sample(self, x_train, y_train, x_pred):

        hf_model_evals_pred = []
        regression_model = []
        if self.regression_type == 'gaussian_process':

            # Fit a GP regression model to approximate p(q_l|q_l-1)
            kernel = Product(RBF(), ConstantKernel()) + WhiteKernel() + ConstantKernel() + DotProduct()
            regression_model = gaussian_process.GaussianProcessRegressor(kernel=kernel)
            regression_model.fit(x_train, y_train)

            # Predict q_l|q_l-1 at all low-fidelity samples
            mu, sigma = regression_model.predict(x_pred, return_std=True)

            # Generate low-fidelity samples from the predictions
            hf_model_evals_pred = np.zeros((mu.shape[0],))
            for i in range(mu.shape[0]):
                hf_model_evals_pred[i] = mu[i] + sigma[i] * np.random.randn()

        else:
            print('Unknown regression model.')
            exit()

        # Save regression model
        self.regression_models.append(regression_model)

        return hf_model_evals_pred

    def get_high_fidelity_samples(self):
        return self.models[-1].model_evals_pred

    def calculate_mc_reference(self):
        self.mc_model = Model(eval_fun=self.models[-1].eval_fun, n_evals=self.models[0].n_evals,
                              n_qoi=self.models[-1].n_qoi, rv_samples=self.models[0].rv_samples,
                              rv_samples_pred=self.models[0].rv_samples, label='MC reference',
                              rv_name=self.models[-1].rv_name)
        self.mc_model.evaluate()
        self.mc_model.set_model_evals_pred(self.mc_model.model_evals)
        self.mc_model.create_distribution()

    def print_stats(self):
        print('')
        print('########### BMFMC statistics ###########\n')
        print('BMFMC mean:\t\t\t\t\t\t%f' % np.mean(self.models[-1].model_evals_pred))
        print('BMFMC std:\t\t\t\t\t\t%f' % np.std(self.models[-1].model_evals_pred))
        print('Low-fidelity mean:\t\t\t\t%f' % np.mean(self.models[0].model_evals_pred))
        print('Low-fidelity std:\t\t\t\t%f' % np.std(self.models[0].model_evals_pred))
        print('\n########################################')
        print('')

    def print_stats_with_mc(self):

        if self.mc_model == 0:
            print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
            exit()

        print('')
        print('########### BMFMC statistics ###########\n')
        print('MC mean:\t\t\t\t\t\t%f' % np.mean(self.mc_model.model_evals_pred))
        print('MC std:\t\t\t\t\t\t\t%f' % np.std(self.mc_model.model_evals_pred))
        print('BMFMC mean:\t\t\t\t\t\t%f' % np.mean(self.models[-1].model_evals_pred))
        print('BMFMC std:\t\t\t\t\t\t%f' % np.std(self.models[-1].model_evals_pred))
        print('Low-fidelity mean:\t\t\t\t%f' % np.mean(self.models[0].model_evals_pred))
        print('Low-fidelity std:\t\t\t\t%f' % np.std(self.models[0].model_evals_pred))
        print('\n########################################')
        print('')

    def plot_results(self):

        # Determine bounds
        xmin = np.min([np.min(self.models[-1].model_evals_pred), np.min(self.models[0].model_evals_pred)])
        xmax = np.max([np.max(self.models[-1].model_evals_pred), np.max(self.models[0].model_evals_pred)])

        for i in range(self.n_models):

            # Plot
            color = 'C' + str(i)
            self.models[i].distribution.plot_kde(fignum=self.fignum, color=color, xmin=xmin, xmax=xmax,
                                                  title='BMFMC - approximate distributions')

        plt.gcf().savefig('pngout/bmfmc_dists.png', dpi=300)
        self.fignum += 1

    def plot_results_with_mc(self):

        if self.mc_model == 0:
            print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
            exit()

        # Determine bounds
        xmin = np.min([np.min(self.mc_model.model_evals_pred), np.min(self.models[0].model_evals_pred)])
        xmax = np.max([np.max(self.mc_model.model_evals_pred), np.max(self.models[0].model_evals_pred)])

        for i in range(self.n_models):

            # Plot
            color = 'C' + str(i)
            self.models[i].distribution.plot_kde(fignum=self.fignum, color=color, xmin=xmin, xmax=xmax,
                                                  title='BMFMC - approximate distributions')

        self.mc_model.distribution.plot_kde(fignum=self.fignum, color='C' + str(self.n_models), linestyle='--',
                                            xmin=xmin, xmax=xmax, title='BMFMC - approximate distributions')

        plt.gcf().savefig('pngout/bmfmc_dists.png', dpi=300)
        self.fignum += 1

    def plot_regression_models(self):

        for i in range(self.n_models-1):

            lf_model = self.models[i]
            hf_model = self.models[i+1]

            regression_model = self.regression_models[i]

            x_train = regression_model.X_train_
            y_train = regression_model.y_train_
            x_pred = lf_model.model_evals_pred
            mu, sigma = regression_model.predict(x_pred, return_std=True)

            # Sort to be able to use the plt.fill
            sort_indices = np.argsort(x_pred, axis=0)
            x_pred = np.squeeze(x_pred[sort_indices])
            y_pred = np.squeeze(mu[sort_indices])
            sigma = np.squeeze(sigma[sort_indices])

            utils.plot_1d_conf(x_pred, y_pred, sigma, num=self.fignum)
            utils.plot_1d_data(x_train, y_train, marker='*', linestyle='', markersize=5, color='k', num=self.fignum,
                               label='Training', title='BMFMC - regression model', xlabel=lf_model.rv_name,
                               ylabel=hf_model.rv_name)

            plt.gcf().savefig('pngout/bmfmc_regression_model_' + str(i+1) + '.png', dpi=300)
            self.fignum += 1

    def plot_joint_densities(self):

        for i in range(self.n_models-1):

            lf_model = self.models[i]
            hf_model = self.models[i+1]

            utils.plot_2d_contour(samples_x=np.squeeze(lf_model.model_evals_pred),
                                  samples_y=np.squeeze(hf_model.model_evals_pred),
                                  num=self.fignum,
                                  title='Approximate joint $p($' + lf_model.rv_name + '$,$' + hf_model.rv_name + '$)$',
                                  xlabel=lf_model.rv_name,
                                  ylabel=hf_model.rv_name)

            plt.gcf().savefig('pngout/bmfmc_joint_dist_' + str(i+1) + '.png', dpi=300)
            self.fignum += 1