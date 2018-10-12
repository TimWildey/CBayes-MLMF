# cbayes-mfmc folder

## MFMC.py
Driver for the Multi-fidelity Monte Carlo framework.

## CBayes.py
Driver for the CBayes framework.

## Distribution.py
A data structure for probability densities.

## Model.py
A data structure for computational models.

## Regression.py
A data structure for regression models.

## utils.py
Plotting functionality.

## models/
Includes some model problems to generate samples from:
* elliptic_pde.py: a single-phase incompressible flow model
* lambda_p.py: a polynomial model

## output/
Figures are stored in here as well as console output.

## convergence_studies/
Some scripts to examine the convergence behavior.

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

* There is a main.py script to run the framework.
* It has a config section to specify the models & methods.
