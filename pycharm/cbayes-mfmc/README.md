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
Includes some model problems to generate/load samples from:
* elliptic_pde.py: a single-phase incompressible flow model
* lambda_p.py: a polynomial model
* linear_elasticity.py: a linear (crystal) elasticity model

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

---

## Examples

### 4.1

* model = 'lambda_p'
* fw_uq_method = 'mfmc'
* n_evals = [20000, 50, 25]
* n_mc_ref = int(2e4)
* training_set_strategy = 'support_covering'
* regression_type = 'gaussian_process'

### 4.2.1

* model = 'elliptic_pde'
* fw_uq_method = 'mfmc'
* n_evals = [10000, 200, 20]
* n_mc_ref = int(5e4)
* training_set_strategy = 'support_covering'
* regression_type = 'heteroscedastic_gaussian_process'

### 4.2.2

* model = 'elliptic_pde_3d'
* fw_uq_method = 'mfmc'
* n_evals = [10000, 1000, 100]
* n_mc_ref = int(5e4)
* training_set_strategy = 'sampling'
* regression_type = 'heteroscedastic_gaussian_process'

### 4.3

* model = 'linear_elasticity'
* fw_uq_method = 'mfmc'
* n_evals = [10000, 100]
* training_set_strategy = 'fixed'
* regression_type = 'gaussian_process'