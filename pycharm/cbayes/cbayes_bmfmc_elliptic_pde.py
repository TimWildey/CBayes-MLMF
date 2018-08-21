import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as GKDE
from scipy.stats import norm
import bmfmc_elliptic_pde
import cbayes_utils as utils
#from matplotlib2tikz import save as tikz_save


# Approximate the pushforward of the prior
q_kde, qvals, lam = bmfmc_elliptic_pde.create_bmfmc_density(n_hf=50, fun=lambda x: np.sin(x))

# Define an observed density and evaluate
obs_loc = 0.7
obs_scale = 0.01
obs_vals = norm.pdf(qvals, loc=obs_loc, scale=obs_scale)

# Compute r
r = np.divide(obs_vals, q_kde(qvals))

# Use rejection sampling for the CBayes posterior
samples_to_keep = utils.rejection_sampling(r)
post_q = qvals[samples_to_keep]
post_lam = lam[samples_to_keep, :]
accept_rate = samples_to_keep.size / lam.shape[0]

# Use KDE to estimate the push-forward of the CBayes posterior
postq_kde = GKDE(post_q)

# Compare the observed and the push-forwards of prior and posterior
qplot = np.linspace(0.6, 0.9, num=100)
postq_kde_plot = postq_kde(qplot)
q_kde_plot = q_kde(qplot)
obs_vals_plot = norm.pdf(qplot, loc=0.7, scale=1.0e-2)

# Plot densities
plt.figure()
oplot = plt.plot(qplot, obs_vals_plot, 'r-', linewidth=4, label="Observed")
prplot = plt.plot(qplot, q_kde_plot, 'b-', linewidth=4, label="PF of prior")
poplot = plt.plot(qplot, postq_kde_plot, 'k--', linewidth=4, label="PF of posterior")
plt.xlim([0.6, 0.9])
plt.xlabel("Quantity of interest")
plt.legend()
plt.gcf().set_figheight(6)
plt.gcf().set_figwidth(6)
plt.gcf().savefig('../pngout/elliptic_pde_bmfmc_cbayes1.png', dpi=300)

# Plot posterior marginal means
plt.figure()
param = np.arange(100)
plt.barh(param, np.mean(post_lam, 0))
plt.xlim([-0.4, 0.4])
plt.xlabel("Mean of marginals of posterior")
plt.ylabel("KL Mode")
plt.gcf().set_figheight(6)
plt.gcf().set_figwidth(6)
plt.gcf().savefig('../pngout/elliptic_pde_bmfmc_cbayes2.png', dpi=300)

# Print some stats
print('')
print('################### CBayes statistics:')
print('Acceptance rate: ' + str(accept_rate))
print('Posterior push-forward mean: ' + str(np.mean(post_q)))
print('Posterior push-forward std: ' + str(np.sqrt(np.var(post_q))))
print('Posterior integral: ' + str(np.mean(r)))
print('Posterior-prior KL: ' + str(np.mean(r * np.log(r))))
qs = np.random.randn(int(1e4)) * obs_scale + obs_loc
qe = norm.pdf(qs, loc=obs_loc, scale=obs_scale)
pe = postq_kde(qs)
print('Posterior-PF-Obs KL: ' + str(np.mean(np.log(qe / pe))))
print('')

# tikz_save('../texout/cbayes_bmfmc_elliptic_pde_fig1.tex')
# tikz_save('../texout/cbayes_bmfmc_elliptic_pde_fig2.tex')
# plt.show()
exit()