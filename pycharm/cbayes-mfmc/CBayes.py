import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

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

        if p_obs.n_dim > 3:
            print('Framework has only been tested with up to 3 QoIs.')
            exit()

        self.p_obs = p_obs
        self.p_prior = p_prior
        self.p_prior_pf = p_prior_pf
        self.p_post = None
        self.p_post_pf = None
        self.r = None
        self.acc_rate = None
        self.acc_idx = None

    # Perform accept/reject sampling on a set of proposal samples using the weights r associated with the set of
    # samples and return the indices idx of the proposal sample set that are accepted.
    def generate_posterior_samples(self):

        # Calculate the weights
        r = np.divide(self.p_obs.kernel_density(np.squeeze(self.p_prior_pf.samples).T) + 1e-10,
                      self.p_prior_pf.kernel_density(np.squeeze(self.p_prior_pf.samples).T))

        # Check against
        check = np.random.uniform(low=0, high=1, size=r.size)

        # Normalize weights
        r_scaled = r / np.max(r)

        # Evaluate criterion
        idx = np.where(r_scaled >= check)[0]

        self.r = r
        self.acc_rate = idx.size / r.shape[0]

        if self.acc_rate < 1.0e-2:
            warnings.warn('Small acceptance rate: %f / %d accepted samples.' % (self.acc_rate, idx.size))

        return self.p_prior.samples[idx], self.p_prior_pf.samples[idx]

    # Create the posterior and its push-forward
    def setup_posterior_and_pf(self):

        # Sample the posterior
        post_samples, post_pf_samples = self.generate_posterior_samples()

        # Create a posterior distribution
        self.p_post = Distribution(samples=post_samples, rv_name=self.p_prior.rv_name,
                                   rv_transform=self.p_prior.rv_transform,
                                   label='Updated', kde=False)

        # Create the posterior push-forward distribution
        self.p_post_pf = Distribution(samples=post_pf_samples, rv_name=self.p_obs.rv_name,
                                      rv_transform=self.p_obs.rv_transform,
                                      label='PF Updated')

    # Get the KL between prior and posterior
    def get_prior_post_kl(self):
        return np.mean(self.r * np.log(self.r))

    # Print a bunch of output diagnostics
    def print_stats(self):

        print('')
        print('########### CBayes statistics ##########')
        print('')

        # The rejection sampling acceptance rate
        print('Acceptance rate:\t\t\t\t%f' % self.acc_rate)

        # The posterior push-forward mean and std
        # (these should match the observed density)
        print('Posterior push-forward mean:\t%s' % self.p_post_pf.mean())
        print('Posterior push-forward std:\t\t%s' % self.p_post_pf.std())

        # The KL between the push-forward of the posterior and the observed density
        # (this should be very close to zero)
        print('Posterior-PF-Obs KL:\t\t\t%f' % self.p_post_pf.calculate_kl_divergence(self.p_obs))

        # The posterior integral
        # (this should be very close to 1.0)
        print('Posterior integral:\t\t\t\t%f' % np.mean(self.r))

        # The KL between posterior and prior (i.e. how informative is the data?)
        # This is done via r / doing KDE for the prior and posterior densities is infeasible when the number of
        # random variables is large.
        print('Posterior-Prior KL:\t\t\t\t%f' % np.mean(self.r * np.log(self.r)))

        print('')
        print('########################################')
        print('')

    # Plot results
    def plot_results(self, model_tag='hf'):

        # Determine bounds
        xmin = np.min([np.min(self.p_prior_pf.samples), np.min(self.p_obs.samples)])
        xmax = np.max([np.max(self.p_prior_pf.samples), np.max(self.p_obs.samples)])

        # Plot
        if self.p_obs.n_dim == 1:
            self.p_prior_pf.plot_kde(color='C0', xmin=xmin, xmax=xmax)
            self.p_obs.plot_kde(color='C1', xmin=xmin, xmax=xmax)
            self.p_post_pf.plot_kde(color='C2', linestyle='--', xmin=xmin, xmax=xmax)

        elif self.p_obs.n_dim == 2:
            sns.kdeplot(self.p_prior_pf.samples[:, 0], self.p_prior_pf.samples[:, 1], shade=True, shade_lowest=False,
                        cmap='Blues', label='PF-initial', color='C0')
            sns.kdeplot(self.p_obs.samples[:, 0], self.p_obs.samples[:, 1], shade=True, shade_lowest=False, cmap='Reds',
                        label='Observed density', color='C3')
            sns.kdeplot(self.p_post_pf.samples[:, 0], self.p_post_pf.samples[:, 1], cmap='Greys', alpha=1.0,
                        label='PF-updated', color='Black')
            plt.legend(loc='upper right')
            plt.xlabel('$Q_1$')
            plt.ylabel('$Q_2$')
        else:
            return

        plt.grid(b=True)
        plt.gcf().savefig('output/cbayes_dists_%s.pdf' % model_tag, dpi=300)

        # Plot some bivariate distributions
        if self.p_obs.n_dim == 2 and model_tag == 'hf':
            self.p_obs.plot_kde()
            plt.grid(b=True)
            plt.gcf().savefig('output/cbayes_dists_obs.pdf', dpi=300)
            plt.clf()
            self.p_post_pf.plot_kde()
            plt.grid(b=True)
            plt.gcf().savefig('output/cbayes_dists_hf_post_pf.pdf', dpi=300)

        plt.clf()

    # Plot posterior
    def plot_posterior(self, fignum=1, color='C0', linestyle='-', label='Posterior', save_fig=False):

        if self.p_obs.n_dim == 1 and self.p_post.n_dim == 1:
            self.p_post.create_kernel_density()
            xmin = np.min(self.p_prior.samples, axis=0)
            xmax = np.max(self.p_prior.samples, axis=0)
            self.p_post.plot_kde(fignum=fignum, color=color, linestyle=linestyle, label=label, xmin=xmin, xmax=xmax)
            if save_fig:
                plt.grid(b=True)
                plt.gcf().savefig('output/cbayes_post_densities.pdf', dpi=300)

        elif self.p_post.n_dim == 2:
            self.p_post.create_kernel_density()
            self.p_post.plot_kde(fignum=fignum, color=color, linestyle=linestyle, label=label)
            if save_fig:
                xmin = np.min(self.p_prior.samples[:, 0], axis=0)
                xmax = np.max(self.p_prior.samples[:, 0], axis=0)
                ymin = np.min(self.p_prior.samples[:, 1], axis=0)
                ymax = np.max(self.p_prior.samples[:, 1], axis=0)
                plt.xlim([xmin, xmax])
                plt.ylim([ymin, ymax])
                plt.grid(b=True)
                plt.gcf().savefig('output/cbayes_post_densities.pdf', dpi=300)
                plt.clf()
