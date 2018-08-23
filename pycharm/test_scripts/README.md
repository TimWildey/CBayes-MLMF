# test_scripts folder

### bmfmc/
Includes the Bayesian multi-fidelity Monte Carlo (BMFMC) scripts to compute approximate densities:
* bmfmc_elliptic_pde.py: application to the single-phase incompressible flow model
* bmfmc_lambda_p.py: application to the polynomial lambda^p problem
* bmfmc_utils.py: some post-processing utils (for plotting etc.)

### cbayes/
Includes the CBayes inversion method:
* cbayes_bmfmc_elliptic_pde.py: CBayes using BMFMC for the single-phase incompressible flow model
* cbayes_bmfmc_lambda_p.py: CBayes using BMFMC for the polynomial lambda^p problem
* cbayes_elliptic_pde.py: CBayes using MC for the single-phase incompressible flow model
* cbayes_lambda_p.py: CBayes using MC for the polynomial lambda^p problem
* cbayes_utils.py: some utils (rejection sampling etc.)

### models/
Includes some model problems (high- and low-fidelity included) to generate samples from:
* elliptic_pde.py: the single-phase incompressible flow model
* lambda_p.py: the polynomial lambda^p problem
* ode_pp.py: a preditor-prey ODE model with 4 random variables

### pngout/
Figures are stored in here as PNGs.

### texout/
Matlabplot2tikz will write .tex files in here (currently not used).

---

## Dependencies

* numpy
* scipy
* matplotlib
* scikit-learn

---

## Usage

* Most .py scripts have a main method with an exemplary use case
* To see if everything is working properly, try to execute cbayes_bmfmc_lambda_p.py. The program should finish after about 20s and some plots should appear in pngout/.