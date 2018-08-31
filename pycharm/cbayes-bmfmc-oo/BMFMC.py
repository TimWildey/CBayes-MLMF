import matplotlib.pyplot as plt
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Product, DotProduct

import utils
from Model import Model


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
        self.adaptive = True
        self.adaptive_tol = 2.0e-3

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
            x_train = []
            adaptive_run = 0

            while self.adaptive:
                adaptive_run += 1

                # 2) Select lower-fidelity model evaluation points and evaluate the next higher-fidelity model
                if len(x_train) == 0:
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
                    self.check_adaptive_convergence(this_dist=hf_model.distribution, previous_dist=previous_dist,
                                                    adaptive_run=adaptive_run)
                    previous_dist = hf_model.distribution

        return self.models[-1].model_evals_pred

    def check_adaptive_convergence(self, this_dist, previous_dist, adaptive_run):

        kl = this_dist.calculate_kl_divergence(previous_dist)

        print('')
        print('Adaptive run %d, KL: %f' % (adaptive_run, kl))

        if kl <= self.adaptive_tol:
            self.adaptive = False
            print('Converged!')
        elif adaptive_run >= 20:
            print('No convergence after 20 runs... aborting.')
            exit()

    def create_training_set(self, lf_model, hf_model, id):

        x_train = []

        if self.training_set_strategy == 'support_covering' or self.training_set_strategy == 'support_covering_adaptive':

            if len(self.regression_models) == id:
                # Create a uniform grid across the support of p(q)
                x_train_linspace = np.linspace(lf_model.model_evals_pred.min(), lf_model.model_evals_pred.max(),
                                               num=hf_model.n_evals)
                x_train_linspace = np.reshape(x_train_linspace, (hf_model.n_evals, hf_model.n_qoi))
            else:
                # Create new points between existing ones
                sorted_xtrain = np.sort(self.regression_models[id].X_train_, axis=0)
                diffs = np.diff(sorted_xtrain, n=1, axis=0)
                diffs = np.reshape(diffs, (np.shape(diffs)[0], hf_model.n_qoi))
                x_train_linspace = sorted_xtrain[:-1, :] + 0.5 * diffs

            n_train = np.shape(x_train_linspace)[0]

            # Find the lower-fidelity samples closest to the grid points and get the corresponding lambdas
            x_train = np.zeros((n_train, hf_model.n_qoi))
            hf_rv_samples = np.zeros((n_train, lf_model.n_random))
            for i in range(n_train):

                # This is only exact for the lowest fidelity regression model
                if id == 0:
                    idx = (np.abs(lf_model.model_evals_pred - x_train_linspace[i, :])).argmin()
                    x_train[i] = lf_model.model_evals_pred[idx]

                # For any other regression model, finding the correct x,y pair is a noisy task.
                # We use the mean predictions of the GP to do that.
                else:
                    regression_model = self.regression_models[id-1]
                    mu = regression_model.predict(self.models[id-1].model_evals_pred, return_std=False)
                    idx = (np.abs(mu - x_train_linspace[i, :])).argmin()
                    x_train[i] = lf_model.eval_fun(lf_model.rv_samples_pred[idx, :])

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
                    x_train[i] = lf_model.eval_fun(lf_model.rv_samples_pred[indices[i], :])
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

    def build_regression_predict_and_sample(self, x_train, y_train, x_pred, id):

        hf_model_evals_pred = []
        regression_model = []

        if self.regression_type == 'gaussian_process':

            # Fit a GP regression model to approximate p(q_l|q_l-1)
            kernel = Product(RBF(), ConstantKernel()) + WhiteKernel() + ConstantKernel() + DotProduct()
            # kernel = Matern() + WhiteKernel()
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
        if len(self.regression_models) == id:
            self.regression_models.append(regression_model)
        elif len(self.regression_models) == id + 1:
            self.regression_models[id] = regression_model
        else:
            print('This is not supposed to happen. Something went very wrong.')
            exit()

        return hf_model_evals_pred

    def get_high_fidelity_samples(self):

        return self.models[-1].model_evals_pred

    def get_low_fidelity_samples(self):

        return self.models[0].model_evals_pred

    def get_mc_samples(self):

        if self.mc_model != 0:

            return self.mc_model.model_evals_pred

        else:
            print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
            exit()

    def get_samples(self):

        samples = np.zeros((self.n_models, self.models[0].n_samples, self.models[0].n_qoi))
        for i in range(self.n_models):
            samples[i, :, :] = self.models[i].model_evals_pred

        return samples

    def calculate_mc_reference(self):

        self.mc_model = Model(eval_fun=self.models[-1].eval_fun, n_evals=self.models[0].n_evals,
                              n_qoi=self.models[-1].n_qoi, rv_samples=self.models[0].rv_samples,
                              rv_samples_pred=self.models[0].rv_samples, label='MC reference',
                              rv_name=self.models[-1].rv_name)
        self.mc_model.evaluate()
        self.mc_model.set_model_evals_pred(self.mc_model.model_evals)
        self.mc_model.create_distribution()

    def print_stats(self, mc=False):

        print('')
        print('########### BMFMC statistics ###########')
        print('')
        if mc and self.mc_model != 0:
            print('MC mean:\t\t\t\t\t\t%f' % self.mc_model.distribution.mean())
            print('MC std:\t\t\t\t\t\t\t%f' % self.mc_model.distribution.std())
            print('')
            print('MC-BMFMC KL:\t\t\t\t\t%f' % self.mc_model.distribution.calculate_kl_divergence(
                self.models[-1].distribution))
        elif mc and self.mc_model == 0:
            print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
            exit()
        print('')
        print('Low-fidelity mean:\t\t\t\t%f' % self.models[0].distribution.mean())
        print('Low-fidelity std:\t\t\t\t%f' % self.models[0].distribution.std())
        print('')
        for i in range(self.n_models - 2):
            print('Mid-%d-fidelity mean:\t\t\t%f' % (int(i + 1), self.models[i + 1].distribution.mean()))
            print('Mid-%d-fidelity std:\t\t\t\t%f' % (int(i + 1), self.models[i + 1].distribution.std()))
            print('')
            kl = self.models[i].distribution.calculate_kl_divergence(self.models[i + 1].distribution)
            print('Relative information gain:\t\t%f' % kl)
            print('')
        print('High-fidelity mean:\t\t\t\t%f' % self.models[-1].distribution.mean())
        print('High-fidelity std:\t\t\t\t%f' % self.models[-1].distribution.std())
        print('')
        kl = self.models[-2].distribution.calculate_kl_divergence(self.models[-1].distribution)
        print('Relative information gain:\t\t%f' % kl)
        print('')
        print('Total information gain:\t\t\t%f' % self.models[0].distribution.calculate_kl_divergence(
            self.models[-1].distribution))
        print('')
        print('########################################')
        print('')

    def plot_results(self, mc=False):

        # Determine bounds
        xmin = np.min([np.min(self.models[-1].model_evals_pred), np.min(self.models[0].model_evals_pred)])
        xmax = np.max([np.max(self.models[-1].model_evals_pred), np.max(self.models[0].model_evals_pred)])

        for i in range(self.n_models):
            # Plot
            color = 'C' + str(i)
            self.models[i].distribution.plot_kde(fignum=self.fignum, color=color, xmin=xmin, xmax=xmax,
                                                 title='BMFMC - approximate distributions')

        if mc and self.mc_model != 0:
            self.mc_model.distribution.plot_kde(fignum=self.fignum, color='k', linestyle='--',
                                                xmin=xmin, xmax=xmax, title='BMFMC - approximate distributions')
        elif mc and self.mc_model == 0:
            print('No Monte Carlo reference samples available. Call calculate_mc_reference() first.')
            exit()

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

        self.mc_model.distribution.plot_kde(fignum=self.fignum, color='k', linestyle='--',
                                            xmin=xmin, xmax=xmax, title='BMFMC - approximate distributions')

        plt.gcf().savefig('pngout/bmfmc_dists.png', dpi=300)
        self.fignum += 1

    def plot_regression_models(self):

        for i in range(self.n_models - 1):
            lf_model = self.models[i]
            hf_model = self.models[i + 1]

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

            if self.n_models > 2:
                plt.gcf().savefig('pngout/bmfmc_regression_model_' + str(i + 1) + '.png', dpi=300)
            else:
                plt.gcf().savefig('pngout/bmfmc_regression_model.png', dpi=300)
            self.fignum += 1

    def plot_joint_densities(self):

        for i in range(self.n_models - 1):
            lf_model = self.models[i]
            hf_model = self.models[i + 1]

            utils.plot_2d_contour(samples_x=np.squeeze(lf_model.model_evals_pred),
                                  samples_y=np.squeeze(hf_model.model_evals_pred),
                                  num=self.fignum,
                                  title='Approximate joint $p($' + lf_model.rv_name + '$,$' + hf_model.rv_name + '$)$',
                                  xlabel=lf_model.rv_name,
                                  ylabel=hf_model.rv_name)

            if self.n_models > 2:
                plt.gcf().savefig('pngout/bmfmc_joint_dist_' + str(i + 1) + '.png', dpi=300)
            else:
                plt.gcf().savefig('pngout/bmfmc_joint_dist.png', dpi=300)
            self.fignum += 1
