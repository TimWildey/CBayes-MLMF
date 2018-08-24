import numpy as np
import matplotlib.pyplot as plt
from Distribution import Distribution


class CBayesPosterior:

    # Class attributes:
    # - p_obs: the observed density
    # - p_prior: the prior
    # - p_prior_pf: the push-forward of the prior
    # - p_post: the posterior
    # - p_post_pf: the push-forward of the posterior
    # - r: the ratio between p_obs and p_prior_pf evaluations
    # - acc_rate: the acceptance rate of the sampling algorithm

    # Constructor
    def __init__(self, p_obs, p_prior, p_prior_pf):

        assert type(p_obs) is Distribution, "p_obs is not of type Distribution: %r" % p_obs
        assert type(p_prior) is Distribution, "p_prior is not of type Distribution: %r" % p_prior
        assert type(p_prior_pf) is Distribution, "p_prior_pf is not of type Distribution: %r" % p_prior_pf

        if p_obs.n_dim > 1:
            print('Framework still lives in a 1-D world.')
            exit()

        self.p_obs = p_obs
        self.p_prior = p_prior
        self.p_prior_pf = p_prior_pf
        self.p_post = []
        self.p_post_pf = []
        self.r = []
        self.acc_rate = []

    # Perform accept/reject sampling on a set of proposal samples using the weights r associated with the set of
    # samples and return the indices idx of the proposal sample set that are accepted.
    def generate_posterior_samples(self):

        # Calculate the weights
        r = np.divide(self.p_obs.kernel_density(np.squeeze(self.p_prior_pf.samples)),
                      self.p_prior_pf.kernel_density(np.squeeze(self.p_prior_pf.samples)))

        # Check against
        check = np.random.uniform(low=0, high=1, size=r.size)

        # Normalize weights
        r_scaled = r / np.max(r)

        # Evaluate criterion
        idx = np.where(r_scaled >= check)[0]

        self.r = r
        self.acc_rate = idx.size / r.shape[0]

        return self.p_prior.samples[idx], self.p_obs.samples[idx]

    # Create the posterior and its push-forward
    def setup_posterior_and_pf(self):

        # Sample the posterior
        post_samples, post_pf_samples = self.generate_posterior_samples()

        # Create a posterior distribution
        self.p_post = Distribution(samples=post_samples, rv_name=self.p_prior.rv_name,
                                   rv_transform=self.p_prior.rv_transform,
                                   label='Posterior', kde=False)

        # Create the posterior push-forward distribution
        self.p_post_pf = Distribution(samples=post_pf_samples, rv_name=self.p_obs.rv_name,
                                      rv_transform=self.p_obs.rv_transform,
                                      label='Posterior-PF')

        return self.p_post, self.p_post_pf

    # Print a bunch of output diagnostics
    def print_stats(self):

        print('')
        print('########### CBayes statistics ##########\n')

        # The rejection sampling acceptance rate
        print('Acceptance rate:\t\t\t\t%f' % self.acc_rate )

        # The posterior push-forward mean and std
        # (these should match the observed density)
        print('Posterior push-forward mean:\t%f' % self.p_post_pf.mean())
        print('Posterior push-forward std:\t\t%f' % self.p_post_pf.std())

        # The KL between the push-forward of the posterior and the observed density
        # (this should be very close to zero)
        print('Posterior-PF-Obs KL:\t\t\t%f' % np.mean(np.log(self.p_obs.kernel_density(np.squeeze(self.p_obs.samples)) /
                                                self.p_post_pf.kernel_density(np.squeeze(self.p_obs.samples)))))

        # The posterior integral
        # (this should be very close to 1.0)
        print('Posterior integral:\t\t\t\t%f' % np.mean(self.r))

        # The KL between posterior and prior (i.e. how informative is the data?)
        # (add a very small number to avoid taking log(0))
        print('Posterior-Prior KL:\t\t\t\t%f' % np.mean(self.r * np.log(self.r + 1e-10)))

        print('\n########################################')
        print('')

    # Plot results
    def plot_results(self, fignum=10):

        # Determine bounds
        xmin = np.min([np.min(self.p_prior_pf.samples), np.min(self.p_obs.samples)])
        xmax = np.max([np.max(self.p_prior_pf.samples), np.max(self.p_obs.samples)])

        # Plot
        self.p_prior_pf.plot_kde(fignum=fignum, color='C0', xmin=xmin, xmax=xmax)
        self.p_obs.plot_kde(fignum=fignum, color='C1', xmin=xmin, xmax=xmax)
        self.p_post_pf.plot_kde(fignum=fignum, color='C2', linestyle='--', xmin=xmin, xmax=xmax,
                                title='CBayes')

        plt.gcf().savefig('pngout/cbayes_dists.png', dpi=300)