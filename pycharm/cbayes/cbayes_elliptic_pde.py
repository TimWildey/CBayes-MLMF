## This code is basically a copy from the jupyter notebook: CBayesTutorial_Example2.ipynb

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as GKDE
from scipy.stats import norm
import elliptic_pde


def rejection_sampling(r):
    # Perform accept/reject sampling on a set of proposal samples using
    # the weights r associated with the set of samples and return
    # the indices idx of the proposal sample set that are accepted.
    N = r.size # size of proposal sample set
    check = np.random.uniform(low=0,high=1,size=N) # create random uniform weights to check r against
    M = np.max(r)
    new_r = r/M # normalize weights
    idx = np.where(new_r>=check)[0] # rejection criterion
    return idx


# Load data
lam, qvals = elliptic_pde.load_data()
lam = np.transpose(lam)

# Define an observed density
obs_vals = norm.pdf(qvals, loc=0.7, scale=1.0e-2)

# Compute the pushforward of the prior
q_kde = GKDE( qvals, 'silverman' )

# Compute the posterior
r = np.divide(obs_vals, q_kde(qvals))

samples_to_keep = rejection_sampling(r)

post_q = qvals[samples_to_keep]
post_lam = lam[:, samples_to_keep]

accept_rate = samples_to_keep.size / lam.shape[0]
print('\n################### CBayes statistics:')
print('Acceptance rate: ' + str(accept_rate))
print('Posterior push-forward mean: ' + str(np.mean(post_q)))
print('Posterior push-forward std: ' + str(np.sqrt(np.var(post_q))))
print('Posterior integral: ' + str(np.mean(r)))
print('Posterior-prior KL: ' + str(np.mean(r * np.log(r))))

# Compare the observed and the pushforwards of prior and posterior
qplot = np.linspace(0.6, 0.9, num=100)

q_kde_plot = q_kde(qplot)
obs_vals_plot = norm.pdf(qplot, loc=0.7, scale=1.0e-2)
postq_kde = GKDE( post_q, 'silverman' )
postq_kde_plot = postq_kde(qplot)

plt.figure()
oplot = plt.plot(qplot,obs_vals_plot, 'r-', linewidth=4, label="Observed")
prplot = plt.plot(qplot,q_kde_plot,'b-', linewidth=4, label="PF of prior")
poplot = plt.plot(qplot,postq_kde_plot,'k--', linewidth=4, label="PF of posterior")
plt.xlim([0.6,0.9])
plt.xlabel("Quantity of interest")
plt.legend()

plt.figure()
param = np.arange(100)
plt.barh(param,np.mean(post_lam,1))
plt.xlim([-0.4,0.4])
plt.xlabel("Mean of marginals of posterior")
plt.ylabel("KL Mode")

plt.show()
exit()