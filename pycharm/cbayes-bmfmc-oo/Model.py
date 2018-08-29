import numpy as np
from Distribution import Distribution


class Model:

    # Class attributes:
    # - eval_fun: the evaluation method (input: random variables, output: QoIs)
    # - rv_samples: evaluation points for model computations
    # - rv_samples_pred: evaluations points for model predictions using regression
    # - n_random: number of random variables
    # - n_samples: number of samples
    # - n_evals: number of specified model evaluations
    # - n_qoi: number of QoIs
    # - model_evals: computed model outputs
    # - model_evals_pred: predicted model outputs from regression
    # - distribution: KDE on the predicted model outputs
    # - label: model label
    # - rv_name: QoI name

    def __init__(self, eval_fun, n_evals, n_qoi, rv_samples=[], rv_samples_pred=[], label='', rv_name=''):

        if len(rv_samples) != 0:
            self.rv_samples = []
            self.set_rv_samples(rv_samples)
        else:
            self.rv_samples = []
            self.n_random = []
            self.n_samples = []

        if len(rv_samples_pred) != 0:
            self.set_rv_samples_pred(rv_samples_pred)
        else:
            self.rv_samples_pred = []

        self.eval_fun = eval_fun
        self.n_evals = n_evals
        self.n_qoi = n_qoi
        self.model_evals = []
        self.model_evals_pred = []
        self.distribution = []
        self.label = label
        self.rv_name = rv_name

    # Evaluate the simulation model at the specified self.rv_samples
    def evaluate(self):

        # If there are no previous model evaluations, create an empty vector
        if len(self.model_evals) == 0:
            self.model_evals = np.zeros((self.n_evals, self.n_qoi))
            for i in range(self.n_evals):
                self.model_evals[i, :] = self.eval_fun(self.rv_samples[i, :])

        # Else, evaluate and update the vector with the new evaluations
        else:
            for i in range(np.shape(self.model_evals)[0], self.n_samples):
                self.model_evals = np.append(self.model_evals,
                                             np.reshape(self.eval_fun(self.rv_samples[i, :]), (1, self.n_qoi)), axis=0)
        return self.model_evals

    # Set the model evaluation points in self.rv_samples
    def set_rv_samples(self, rv_samples):

        # If there are no rv_samples, create the vector
        if len(self.rv_samples) == 0:
            self.rv_samples = rv_samples
            self.n_samples = np.shape(rv_samples)[0]
            self.n_random = np.shape(rv_samples)[1]

        # Else, update the vector with the given vector
        else:
            for row in rv_samples:
                # if not np.equal(row, self.rv_samples).all(axis=1).any():
                self.rv_samples = np.append(self.rv_samples, np.reshape(row, (1, self.n_random)), axis=0)

            self.n_samples = np.shape(self.rv_samples)[0]

    # Set the model evaluation points for prediction (those will be the same of any fidelity model)
    def set_rv_samples_pred(self, rv_samples_pred):

        self.rv_samples_pred = rv_samples_pred

    # Set the model predictions at the specified self.rv_samples_pred
    def set_model_evals_pred(self, model_evals_pred):

        self.model_evals_pred = model_evals_pred

    # Create a Distribution object from the model predictions
    def create_distribution(self):

        self.distribution = Distribution(samples=self.model_evals_pred, rv_name=self.rv_name, label=self.label)
