# cbayes-bmfmc-oo folder

## BMFMC.py
* Driver for the Bayesian multi-fidelity Monte Carlo framework.

## CBayes.py
* Driver for the CBayes framework.

## Distribution.py
* A distribution data structure.

## Model.py
* A data structure for models.

## utils.py
* Plotting functionality.

## models/
Includes some model problems to generate samples from:
* elliptic_pde.py: the single-phase incompressible flow model
* lambda_p.py: the polynomial lambda^p problem
* ode_pp.py: a preditor-prey ODE model with 4 random variables

## pngout/
Figures are stored in here as PNGs.

---

# Dependencies

* numpy
* scipy
* matplotlib
* scikit-learn
* seaborn
* pandas
* gp_extras (https://github.com/jmetzen/gp_extras/) for the heteroscedastic GP

---

## Usage

* There is one main.py script which can be run.
* It has a config section to specify the models & methods.
