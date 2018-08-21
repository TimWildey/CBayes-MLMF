# PyCharm project folder

## bmfmc/
Includes the Bayesian multi-fidelity Monte Carlo (BMFMC) scripts to compute approximate densities:
* bmfmc_elliptic_pde.py: application to the single-phase incompressible flow model
* bmfmc_lambda_p.py: application to the polynomial lambda^p problem
* bmfmc_utils.py: some post-processing utils (for plotting etc.)

## cbayes/
Includes the CBayes inversion method:
* cbayes_bmfmc_elliptic_pde.py: CBayes using BMFMC for the single-phase incompressible flow model
* cbayes_bmfmc_lambda_p.py: CBayes using BMFMC for the polynomial lambda^p problem
* cbayes_elliptic_pde.py: CBayes using MC for the single-phase incompressible flow model
* cbayes_lambda_p.py: CBayes using MC for the polynomial lambda^p problem
* cbayes_utils.py: some utils (rejection sampling etc.)

## models/
Includes some model problems (high- and low-fidelity included) to generate samples from:
* elliptic_pde.py: the single-phase incompressible flow model
* lambda_p.py: the polynomial lambda^p problem

## pngout/
Figures are stored in here as PNGs.

## texout/
Matlabplot2tikz will write .tex files in here (currently not used).

---

# Dependencies

* numpy
* scipy
* matplotlib
* scikit-learn

---

# Usage

* Most .py scripts have a main method with an exemplary use case
* To see if everything is working properly, try to execute cbayes_bmfmc_lambda_p.py. The program should finish after about 20s and some plots should appear in pngout/.

---

# Literature

## CBayes
* T. Butler, J. Jakeman, and T. Wildey, “Combining Push-Forward Measures and Bayes’ Rule to Construct Consistent Solutions to Stochastic Inverse Problems,” SIAM Journal on Scientific Computing, vol. 40, no. 2, pp. A984–A1011, Jan. 2018.
* T. Butler, J. Jakeman, and T. Wildey, “Convergence of Probability Densities using Approximate Models for Forward and Inverse Problems in Uncertainty Quantification,” 2018, arXiv:1807.00375 [math.NA]

## BMFMC
* P.-S. Koutsourelakis, “Accurate Uncertainty Quantification Using Inaccurate Computational Models,” SIAM Journal on Scientific Computing, vol. 31, no. 5, pp. 3274–3300, Jan. 2009.
* J. Biehler, M. W. Gee, and W. A. Wall, “Towards efficient uncertainty quantification in complex and large-scale biomechanical problems based on a Bayesian multi-fidelity scheme,” Biomechanics and Modeling in Mechanobiology, vol. 14, no. 3, pp. 489–513, Jun. 2015.
* A. Quaglino, S. Pezzuto, P.-S. Koutsourelakis, A. Auricchio, and R. Krause, “Fast uncertainty quantification of activation sequences in patient-specific cardiac electrophysiology meeting clinical time constraints: Fast uncertainty quantification in cardiac electrophysiology,” International Journal for Numerical Methods in Biomedical Engineering, Mar. 2018.

