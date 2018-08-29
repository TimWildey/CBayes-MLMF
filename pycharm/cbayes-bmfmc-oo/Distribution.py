import numpy as np
import utils
from scipy.stats import gaussian_kde as gkde
from scipy.integrate import trapz


class Distribution:

    # Class attributes:
    # - rv_name: random variable name(s)
    # - label: distribution label
    # - kernel_density: the KDE density estimate
    # - samples: distribution samples that were used for the KDE with shape (n_samples, dist_dim)
    # - n_samples: number of samples
    # - n_dim: number of random variables
    # - rv_transform: transformation to the QoI space (all distributions must have full support; in most cases
    #   this will be a positivity transform such as x = exp(x))

    # Constructor
    def __init__(self, samples, rv_name='$x$', label='Distribution over x', rv_transform=lambda x: x, kde=True):
        self.rv_name = rv_name
        self.label = label
        self.samples = samples
        self.n_samples = np.shape(samples)[0]
        self.n_dim = np.shape(samples)[1]
        self.rv_transform = rv_transform

        if self.n_dim < 3 and kde:
            self.kernel_density = gkde(np.squeeze(samples))
        elif kde:
            print('Attempting KDE in %d dimensions. Aborting.' % self.n_dim)
            exit()
        else:
            self.kernel_density = []

    # Generate samples
    def generate_samples(self, n_samples):
        indices = np.random.choice(range(self.samples.shape[0]), size=n_samples, replace=True)
        return self.samples[indices]

    # Estimate the distribution sample mean
    def mean(self):
        return np.mean(self.samples, 0)

    # Estimate the distribution sample variance
    def var(self):
        return np.var(self.samples, 0)

    # Estimate the distribution sample standard deviation
    def std(self):
        return np.std(self.samples, 0)

    # Estimate the KL divergence between q and p
    # (KL [q : p] = \int ( \log(q(x)) - \log(p(x)) ) q(x) dx)
    def calculate_kl_divergence(self, p):
        q = self.kernel_density(np.squeeze(self.samples))
        q += 1e-10
        p = p.kernel_density(np.squeeze(self.samples))
        p += 1e-10
        kl = np.mean(np.log(np.divide(q, p)))
        return kl

    # Estimate the KL divergence between q and p
    # (KL [q : p] = \int ( \log(q(x)) - \log(p(x)) ) q(x) dx)
    def calculate_kl_divergence_alt(self, p):
        n_samples = self.n_samples
        x_min = min([np.min(self.samples), np.min(p.samples)])
        x_max = max([np.max(self.samples), np.max(p.samples)])
        x = np.linspace(x_min, x_max, n_samples)
        y1 = self.kernel_density(np.squeeze(x))
        y2 = p.kernel_density(np.squeeze(x))
        y1 += 1e-10
        y2 += 1e-10
        kl_arg = np.multiply(y1, np.log(y1) - np.log(y2))
        kl = trapz(kl_arg, x)
        return kl

    # Plot the KDE density
    def plot_kde(self, fignum, color='C0', linestyle='-', xmin=0.0, xmax=1.0, title=''):
        utils.plot_1d_kde(qkde=self.kernel_density, xmin=xmin, xmax=xmax, label=self.label, linestyle=linestyle,
                          num=fignum, xlabel=self.rv_name, ylabel='$p($' + self.rv_name + '$)$', color=color,
                          title=title)

    # Plot the sample histogram
    def plot_histogram(self, fignum):
        utils.plot_1d_hist(samples=self.samples, num=fignum, xlabel=self.rv_name, ylabel='$p($' + self.rv_name + '$)$')
