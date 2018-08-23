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
            self.set_rv_samples(rv_samples)
        else:
            self.rv_samples = rv_samples
            self.n_random = []
            self.n_samples = []

        if len(rv_samples_pred) != 0:
            self.set_rv_samples_pred(rv_samples_pred)
        else:
            self.rv_samples_pred = rv_samples_pred

        self.eval_fun = eval_fun
        self.n_evals = n_evals
        self.n_qoi = n_qoi
        self.model_evals = []
        self.model_evals_pred = []
        self.distribution = []
        self.label = label
        self.rv_name = rv_name

    def evaluate(self):

        self.model_evals = np.zeros((self.n_evals, self.n_qoi))
        for i in range(self.n_evals):
            self.model_evals[i, :] = self.eval_fun(self.rv_samples[i, :])

        return self.model_evals

    def set_rv_samples(self, rv_samples):

        self.rv_samples = rv_samples
        self.n_samples = np.shape(rv_samples)[0]
        self.n_random = np.shape(rv_samples)[1]

    def set_rv_samples_pred(self, rv_samples_pred):

        self.rv_samples_pred = rv_samples_pred

    def set_model_evals_pred(self, model_evals_pred):

        self.model_evals_pred = model_evals_pred

    def create_distribution(self):

        self.distribution = Distribution(samples=self.model_evals_pred, rv_name=self.rv_name, label=self.label)
