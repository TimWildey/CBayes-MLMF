import numpy as np
from scipy.stats import norm  # The standard Normal distribution
from scipy.stats import gaussian_kde as gkde  # A standard kernel density estimator
import matplotlib.pyplot as plt
import bmfmc_lambda_p
import cbayes_utils as utils
# from matplotlib2tikz import save as tikz_save


# Approximate the pushforward of the prior
q_kde, qvals, lam = bmfmc_lambda_p.create_bmfmc_density(n_lf=10000, n_hf=20, p_lf=3, p_hf=5)

# Evaluate the observed density on the QoI sample set
obs_loc = 0.25
obs_scale = 0.1
obs_vals = norm.pdf(qvals, loc=obs_loc, scale=obs_scale)

# Calculate r
r = np.divide(obs_vals, q_kde(qvals))

# Use rejection sampling for the CBayes posterior
samples_to_keep = utils.rejection_sampling(r)
post_q = qvals[samples_to_keep]
post_lam = lam[samples_to_keep]
accept_rate = samples_to_keep.size / lam.shape[0]

# Compute normalizing constants and use rejection sampling for the SBayes posterior
C = np.mean(obs_vals)
sbayes_r = obs_vals / C
sbayes_samples_to_keep = utils.rejection_sampling(sbayes_r)
sbayes_post_q = qvals[sbayes_samples_to_keep]
sbayes_post_lam = lam[sbayes_samples_to_keep]

# Use KDE to estimate the push-forward of the CBayes and SBayes posterior
postq_kde = gkde(post_q)
sb_postq_kde = gkde(sbayes_post_q)

# Compare the observed and the push-forwards of prior, CBayes posterior and SBayes posterior
plt.figure()
qplot = np.linspace(-1, 1, num=100)
obs_vals_plot = norm.pdf(qplot, loc=obs_loc, scale=obs_scale)
oplot = plt.plot(qplot, obs_vals_plot, 'r-', linewidth=3, label="Observed")
prplot = plt.plot(qplot, q_kde(qplot), 'b-', linewidth=3, label="PF of prior")
poplot = plt.plot(qplot, postq_kde(qplot), 'k--', linewidth=3, label="PF of CBayes posterior")
sb_poplot = plt.plot(qplot, sb_postq_kde(qplot), 'g--', linewidth=3, label="PF of SBayes posterior")
plt.xlim([-1, 1]), plt.xlabel("Quantity of interest"), plt.legend()
plt.gcf().set_figheight(6)
plt.gcf().set_figwidth(6)
plt.gcf().savefig('../pngout/lambda_p_bmfmc_cbayes.png', dpi=300)

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

# tikz_save('../texout/cbayes_bmfmc_lambda_p_fig1.tex')
# tikz_save('../texout/cbayes_bmfmc_lambda_p_fig2.tex')
# plt.show()
exit()
