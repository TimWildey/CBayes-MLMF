## This code is basically a copy from the jupyter notebook: CBayesTutorial_Example1.ipynb

# The libraries we will use
import numpy as np
from scipy.stats import norm  # The standard Normal distribution
from scipy.stats import gaussian_kde as GKDE  # A standard kernel density estimator
import matplotlib.pyplot as plt


def lambdap(lam, p):
    q = lam ** p
    return q


def rejection_sampling(r):
    n = r.size  # size of proposal sample set
    check = np.random.uniform(low=0, high=1, size=n)  # create random uniform weights to check r against
    m = np.max(r)
    new_r = r / m  # normalize weights
    idx = np.where(new_r >= check)[0]  # rejection criterion
    return idx


# Approximate the pushforward of the prior
N = int(1E4)  # number of samples from prior
lam = np.random.uniform(low=-1, high=1, size=N)  # sample set of the prior

# Evaluate the two different QoI maps on this prior sample set
qvals_linear = lambdap(lam, 1)  # Evaluate lam^1 samples
qvals_nonlinear = lambdap(lam, 5)  # Evaluate lam^5 samples

# Estimate push-forward densities for each QoI
q_linear_kde = GKDE(qvals_linear)
q_nonlinear_kde = GKDE(qvals_nonlinear)

# Evaluate the observed density on the QoI sample set and then compute r
obs_vals_linear = norm.pdf(qvals_linear, loc=0.25, scale=0.1)
obs_vals_nonlinear = norm.pdf(qvals_nonlinear, loc=0.25, scale=0.1)

r_linear = np.divide(obs_vals_linear, q_linear_kde(qvals_linear))
r_nonlinear = np.divide(obs_vals_nonlinear, q_nonlinear_kde(qvals_nonlinear))

# Use rejection sampling for the CBayes posterior
samples_to_keep_linear = rejection_sampling(r_linear)
post_q_linear = qvals_linear[samples_to_keep_linear]
post_lam_linear = lam[samples_to_keep_linear]

samples_to_keep_nonlinear = rejection_sampling(r_nonlinear)
post_q_nonlinear = qvals_nonlinear[samples_to_keep_nonlinear]
post_lam_nonlinear = lam[samples_to_keep_nonlinear]

# compute normalizing constants
C_linear = np.mean(obs_vals_linear)
C_nonlinear = np.mean(obs_vals_nonlinear)

sbayes_r_linear = obs_vals_linear / C_linear
sbayes_r_nonlinear = obs_vals_nonlinear / C_nonlinear

sbayes_samples_to_keep_linear = rejection_sampling(sbayes_r_linear)
sbayes_post_q_linear = qvals_linear[sbayes_samples_to_keep_linear]
sbayes_post_lam_linear = lam[sbayes_samples_to_keep_linear]

sbayes_samples_to_keep_nonlinear = rejection_sampling(sbayes_r_nonlinear)
sbayes_post_q_nonlinear = qvals_nonlinear[sbayes_samples_to_keep_nonlinear]
sbayes_post_lam_nonlinear = lam[sbayes_samples_to_keep_nonlinear]

# Compare the observed and the pushforwards of prior, Cbayes posterior and Sbayes posterior
plt.figure()
qplot = np.linspace(-1, 1, num=100)
obs_vals_plot = norm.pdf(qplot, loc=0.25, scale=0.1)

postq_lin_kde = GKDE(post_q_linear)
sb_postq_lin_kde = GKDE(sbayes_post_q_linear)

oplot = plt.plot(qplot, obs_vals_plot, 'r-', linewidth=4, label="Observed")
prplot = plt.plot(qplot, q_linear_kde(qplot), 'b-', linewidth=4, label="PF of prior")
poplot = plt.plot(qplot, postq_lin_kde(qplot), 'k--', linewidth=4, label="PF of CBayes posterior")
sb_poplot = plt.plot(qplot, sb_postq_lin_kde(qplot), 'g--', linewidth=4, label="PF of SBayes posterior")

plt.xlim([-1, 1]), plt.xlabel("Quantity of interest"), plt.legend()

# Compare the observed and the pushforwards of prior, Cbayes posterior and Sbayes posterior
plt.figure()
qplot = np.linspace(-1, 1, num=100)
obs_vals_plot = norm.pdf(qplot, loc=0.25, scale=0.1)

postq_nl_kde = GKDE(post_q_nonlinear)
sb_postq_nl_kde = GKDE(sbayes_post_q_nonlinear)

oplot = plt.plot(qplot, obs_vals_plot, 'r-', linewidth=4, label="Observed")
prplot = plt.plot(qplot, q_nonlinear_kde(qplot), 'b-', linewidth=4, label="PF of prior")
poplot = plt.plot(qplot, postq_nl_kde(qplot), 'k--', linewidth=4, label="PF of Cbayes posterior")
sb_poplot = plt.plot(qplot, sb_postq_nl_kde(qplot), 'g--', linewidth=4, label="PF of SBayes posterior")

plt.xlim([-1, 1]), plt.xlabel("Quantity of interest"), plt.legend()

accept_rate = samples_to_keep_linear.size / lam.shape[0]
print('Acceptance rate: ' + str(accept_rate))

print(np.mean(post_q_linear))
print(np.sqrt(np.var(post_q_linear)))
print(np.mean(r_linear))
print(np.mean(r_linear * np.log(r_linear)))

print(np.mean(post_q_nonlinear))
print(np.sqrt(np.var(post_q_nonlinear)))
print(np.mean(r_nonlinear))
print(np.mean(r_nonlinear * np.log(r_nonlinear)))

plt.show()
exit()