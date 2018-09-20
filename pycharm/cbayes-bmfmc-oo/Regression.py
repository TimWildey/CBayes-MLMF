import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel
from sklearn.cluster import KMeans
from gp_extras.kernels import HeteroscedasticKernel


class Regression:

    # Class attributes:
    # - x_train: low-fidelity evaluations
    # - y_train: corresponding high-fidelity evaluations
    # - x_pred: low-fidelity points, where the high-fidelity solution is predicted
    # - regression_type: regression model type
    # - training_set_strategy:
    # - mu: predicted means
    # - sigma: predicted standard deviations

    # Constructor
    def __init__(self, regression_type, training_set_strategy):

        self.x_train = None
        self.y_train = None
        self.x_pred = None
        self.regression_type = regression_type
        self.training_set_strategy = training_set_strategy
        self.mu = None
        self.sigma = None

    # Create the regression model and make high-fidelity predictions
    def build_regression_predict_and_sample(self):

        hf_model_evals_pred = None

        if self.regression_type == 'gaussian_process':

            # Fit a GP regression model to approximate p(q_l|q_l-1)
            kernel = ConstantKernel(1.0, (1e-10, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel() + ConstantKernel()
            # kernel = Matern() + WhiteKernel()
            regression_model = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
            regression_model.fit(self.x_train, self.y_train)

            # Predict q_l|q_l-1 at all low-fidelity samples
            self.mu, self.sigma = regression_model.predict(self.x_pred, return_std=True)

            # Generate high-fidelity samples from the predictions
            hf_model_evals_pred = np.zeros((self.mu.shape[0], self.mu.shape[1]))
            for i in range(self.mu.shape[0]):
                hf_model_evals_pred[i, :] = self.mu[i, :] + self.sigma[i] * np.random.randn(1, self.mu.shape[1])

        elif self.regression_type == 'decoupled_gaussian_processes':

            # Fit a GP regression model to approximate p(q_l|q_l-1)
            kernel = ConstantKernel(1.0, (1e-10, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel() + ConstantKernel()

            hf_model_evals_pred = np.zeros((self.x_pred.shape[0], self.x_pred.shape[1]))
            regression_model = []
            self.mu = np.zeros(self.x_pred.shape)
            self.sigma = np.zeros(self.x_pred.shape)
            for k in range(self.x_train.shape[1]):

                regression_model.append(gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-10))
                regression_model[k].fit(np.expand_dims(self.x_train[:, k], axis=1), self.y_train[:, k])

                # Predict q_l|q_l-1 at all low-fidelity samples
                self.mu[:, k], self.sigma[:, k] = regression_model[k].predict(np.expand_dims(self.x_pred[:, k], axis=1),
                                                                              return_std=True)

                # Generate high-fidelity samples from the predictions
                for i in range(self.mu.shape[0]):
                    hf_model_evals_pred[i, k] = self.mu[i, k] + self.sigma[i, k] * np.random.randn()

        elif self.regression_type == 'heteroscedastic_gaussian_process':

            # Fit a heteroscedastic GP regression model with spatially varying noise to approximate p(q_l|q_l-1)
            # See here for more info: https://github.com/jmetzen/gp_extras/
            prototypes = KMeans(n_clusters=5).fit(self.x_train).cluster_centers_
            kernel = ConstantKernel(1.0, (1e-10, 1000)) * RBF(1.0, (0.01, 100.0)) + HeteroscedasticKernel.construct(
                prototypes, 1e-3, (1e-10, 50.0), gamma=5.0, gamma_bounds="fixed")
            regression_model = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
            regression_model.fit(self.x_train, self.y_train)

            # Predict q_l|q_l-1 at all low-fidelity samples
            self.mu, self.sigma = regression_model.predict(self.x_pred, return_std=True)

            # Generate high-fidelity samples from the predictions
            hf_model_evals_pred = np.zeros((self.mu.shape[0], self.mu.shape[1]))
            for i in range(self.mu.shape[0]):
                hf_model_evals_pred[i, :] = self.mu[i, :] + self.sigma[i] * np.random.randn(1, self.mu.shape[1])

        elif self.regression_type == 'decoupled_heteroscedastic_gaussian_process':

            # Fit a heteroscedastic GP regression model with spatially varying noise to approximate p(q_l|q_l-1)
            # See here for more info: https://github.com/jmetzen/gp_extras/
            hf_model_evals_pred = np.zeros((self.x_pred.shape[0], self.x_pred.shape[1]))
            regression_model = []
            self.mu = np.zeros(self.x_pred.shape)
            self.sigma = np.zeros(self.x_pred.shape)
            for k in range(self.x_train.shape[1]):
                prototypes = KMeans(n_clusters=5).fit(np.expand_dims(self.x_pred[:, k], axis=1)).cluster_centers_
                kernel = ConstantKernel(1.0, (1e-10, 1000)) * RBF(1.0, (0.01, 100.0)) + HeteroscedasticKernel.construct(
                    prototypes, 1e-3, (1e-10, 50.0), gamma=5.0, gamma_bounds="fixed")

                regression_model.append(gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-10))
                regression_model[k].fit(np.expand_dims(self.x_train[:, k], axis=1), self.y_train[:, k])

                # Predict q_l|q_l-1 at all low-fidelity samples
                self.mu[:, k], self.sigma[:, k] = regression_model[k].predict(np.expand_dims(self.x_pred[:, k], axis=1),
                                                                              return_std=True)

                # Generate high-fidelity samples from the predictions
                for i in range(self.mu.shape[0]):
                    hf_model_evals_pred[i, k] = self.mu[i, k] + self.sigma[i, k] * np.random.randn()

        else:
            print('Unknown regression model %s.' % self.regression_type)
            exit()

        return hf_model_evals_pred

    # Create the training set for the regression
    def create_training_set(self, lf_model, hf_model, id, regression_models):

        x_train = None
        adaptive = True

        if self.training_set_strategy == 'fixed':

            # Find lf_rv_samples corresponding to hf_rv_samples
            x_train = np.zeros((hf_model.n_evals, hf_model.n_qoi))
            for i in range(hf_model.n_evals):
                idx = (np.linalg.norm(lf_model.rv_samples - hf_model.rv_samples[i, :], axis=1, ord=1)).argmin()
                x_train[i, :] = lf_model.model_evals[idx, :]

            adaptive = False

        elif self.training_set_strategy in ['support_covering', 'support_covering_adaptive']:

            if len(regression_models) == id:
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

                elif hf_model.n_qoi is 3:
                    abc_num = int(np.power(hf_model.n_evals, 1. / 3))
                    hf_model.n_evals = abc_num ** 3
                    a = np.linspace(lf_model.model_evals_pred[:, 0].min(), lf_model.model_evals_pred[:, 0].max(),
                                    num=abc_num)
                    b = np.linspace(lf_model.model_evals_pred[:, 1].min(), lf_model.model_evals_pred[:, 1].max(),
                                    num=abc_num)
                    c = np.linspace(lf_model.model_evals_pred[:, 2].min(), lf_model.model_evals_pred[:, 2].max(),
                                    num=abc_num)
                    aa, bb, cc = np.meshgrid(a, b, c)
                    x_train_linspace = np.reshape(np.vstack([aa, bb, cc]), (3, hf_model.n_evals)).T

                else:
                    print('Support covering strategy only available for up to 3 QoIs. Use sampling.')
                    exit()

            else:
                # Create new points between existing ones
                if hf_model.n_qoi is 1:
                    sorted_xtrain = np.sort(regression_models[id].x_train, axis=0)
                    diffs = np.diff(sorted_xtrain, n=1, axis=0)
                    diffs = np.reshape(diffs, (np.shape(diffs)[0], hf_model.n_qoi))
                    x_train_linspace = sorted_xtrain[:-1, :] + 0.5 * diffs

                else:
                    print('Support covering adaptive strategy only available for 1 QoI. Use sampling_adaptive.')
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
                    regression_model = regression_models[id - 1]

                    # This hack is necessary if using the decoupled GPs
                    if isinstance(regression_model, list):
                        mu = np.zeros((regression_model.x_pred.shape[0], hf_model.n_qoi))
                        for k in range(len(regression_model)):
                            mu[:, k] = regression_model[k].mu
                    else:
                        mu = regression_model.mu

                    idx = (np.linalg.norm(mu - x_train_linspace[i, :], axis=1, ord=1)).argmin()
                    x_train[i, :] = lf_model.eval_fun(lf_model.rv_samples_pred[idx, :])

                hf_rv_samples[i, :] = lf_model.rv_samples_pred[idx, :]

            # Assign model evaluation points to the high-fidelity model
            hf_model.set_rv_samples(hf_rv_samples)

            if self.training_set_strategy == 'support_covering':
                # No adaptivity
                adaptive = False

        elif self.training_set_strategy in ['sampling', 'sampling_adaptive']:

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
                adaptive = False

        else:
            print('Unknown training set selection strategy.')
            exit()

        return x_train, adaptive