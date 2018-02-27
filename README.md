# Fast Bayesian nonnegative matrix factorisation and tri-factorisation
This project contains an implementation of the non-negative matrix factorisation and tri-factorisation models presented in the paper [**Fast Bayesian nonnegative matrix factorisation and tri-factorisation**](https://arxiv.org/abs/1610.08127), accepted for the NIPS 2016 Workshop on Advances in Approximate Bayesian Inference.
For both models we implement four different inference methods: Gibbs sampling, variational Bayesian inference, iterated conditional modes, and non-probabilistic inference.
We furthermore provide all datasets used (including the preprocessing scripts), and Python scripts for experiments.

An extended version of this project, with more experiments and automatic model selection (using automatic relevance determination) can be found [here](https://github.com/ThomasBrouwer/BNMTF_ARD); paper [here](https://arxiv.org/abs/1707.05147).

#### Paper abstract
We present a fast variational Bayesian algorithm for performing non-negative matrix factorisation and tri-factorisation. We show that our approach achieves faster convergence per iteration and timestep (wall-clock) than Gibbs sampling and non-probabilistic approaches, and do not require additional samples to estimate the posterior. We show that in particular for matrix tri-factorisation convergence is difficult, but our variational Bayesian approach offers a fast solution, allowing the tri-factorisation approach to be used more effectively.

#### Authors
Thomas Brouwer, Jes Frellsen, Pietro Lio'. Contact: tab43@cam.ac.uk / thomas.a.brouwer@gmail.com.

## Installation 
If you wish to use the matrix factorisation models, or replicate the experiments, follow these steps. Please ensure you have Python 2.7 (3 is currently not supported). 
1. Clone the project to your computer, by running `git clone https://github.com/ThomasBrouwer/BNMTF.git` in your command line.
2. In your Python script, add the project to your system path using the following lines.  
   
   ``` 
   project_location = "/path/to/folder/containing/project/"
   import sys
   sys.path.append(project_location) 
   ```
   For example, if the path to the project is /johndoe/projects/BNMTF/, use `project_location = /johndoe/projects/`.
   If you intend to rerun some of the paper's experiments, those scripts automatically add the correct path.
3. You can now import the models in your code, e.g.
```
from BNMTF.code.models.nmf_np import NMF
model = NMF(R=numpy.ones((4,3)), M=numpy.ones((4,3)), K=2)
model.initialise()
model.train(iterations=10)
```

## Examples
You can find good examples of the models running on data in the [convergence experiment on the toy data](./experiments/experiments_toy/convergence/), e.g. [nonnegative matrix factorisation with Gibbs sampling](./experiments/experiments_toy/convergence/nmf_gibbs.py).


## Citation
If this project was useful for your research, please consider citing the [extended paper](https://arxiv.org/abs/1707.05147),
> Thomas Brouwer, Jes Frellsen, and Pietro Lió (2017). Comparative Study of Inference Methods for Bayesian Nonnegative Matrix Factorisation. Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2017).
```
@inproceedings{Brouwer2017b,
	author = {Brouwer, Thomas and Frellsen, Jes and Li\'{o}, Pietro},
	booktitle = {Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)},
	title = {{Comparative Study of Inference Methods for Bayesian Nonnegative Matrix Factorisation}},
	year = {2017}
}
```

## Project structure
<details>
<summary>Click here to find a description of the different folders and files available in this repository.</summary>

<br>

### /code/
Python code, for the models, cross-validation methods, and model selection.

**/models/**: Python classes for the BNMF and BNMTF models: Gibbs sampling, Variational Bayes, Iterated Conditional Modes, and non-probabilistic versions.
- **/distributions/** - Contains code for obtaining draws of the exponential, Gaussian, and Truncated Normal distributions. Also has code for computing the expectation and variance of these distributions.
- **/kmeans/** - Contains a class for performing K-means clustering on a matrix, when some of the values are unobserved. From my [other Github project](https://github.com/ThomasBrouwer/kmeans_missing).
- **bnmf_gibbs.py** - Implementation of Gibbs sampler for Bayesian non-negative matrix factorisation (BNMF), extended to take into account missing values. Initially introduced by Schmidt et al. 2009.
- **bnmf_vb.py** - Implementation of our variational Bayesian inference for BNMF.
- **nmf_icm.py** - Implementation of Iterated Conditional Modes NMF algorithm (MAP inference). Initially introduced by Schmidt et al. 2009.
- **nmf_np.py** - Implementation of non-probabilistic NMF (Lee and Seung 2001).
- **bnmtf_gibbs.py** - Implementation of our Gibbs sampler for Bayesian non-negative matrix tri-factorisation (BNMTF).
- **bnmtf_vb.py** - Implementation of our variational Bayesian inference for BNMTF.
- **nmtf_icm.py** - Implementation of Iterated Conditional Modes NMTF algorithm (MAP inference).
- **nmtf_np.py** - Implementation of non-probabilistic NMTF, introduced by Yoo and Choi 2009.

**/grid_search/**: Classes for doing model selection on the Bayesian NMF and NMTF models, and for doing cross-validation with model selection. We can minimise or maximise the MSE, ELBO, AIC, BIC, log likelihood.
- **line_search_bnmf.py** - Line search for the BNMF models, trying out different values for K.
- **line_search_cross_validation** - Class for measuring cross-validation performance, with line search to choose K, for the BNMF models.
- **grid_search_bnmtf.py** - Full grid search for the BNMTF models, trying all combinations of values for K and L in a given range.
- **greedy_search_bnmtf.py** - Greedy grid search for the BNMTF models, as described in NIPS workshop paper.
- **greedy_search_cross_validation.py** - Class for measuring cross-validation performance, with greedy search, for the BNMTF models.
- **matrix_cross_validation.py** - Class for finding the best value of K for any of the models (Gibbs, VB, ICM, NP), using cross-validation.
- **parallel_matrix_cross_validation.py** - Same as matrix_cross_validation.py but P folds are ran in parallel.
- **nested_matrix_cross_validation.py** - Class for measuring cross-validation performance, with nested cross-validation to choose K, used for non-probabilistic NMF and NMTF.
- **mask.py** - Contains methods for splitting data into training and test folds.

### /data_toy/
Contains the toy data, and methods for generating toy data.
- **/bnmf/** - Generate toy data using **generate_bnmf.py**, giving files **U.txt**, **V.txt**, **R.txt**, **R_true.txt** (no noise), **M.txt**.
- **/bnmtf/** - Generate toy data using **generate_bnmtf.py**, giving files **F.txt**, **S.txt**, **G.txt**, **R.txt**, **R_true.txt** (no noise), **M.txt**.

### /data_drug_sensitivity/
Contains the drug sensitivity datasets (GDSC IC50, CCLE IC50, CCLE EC50, CTRP EC50).
- **/gdsc/**, **/ccle/** - The datasets. We obtained these from the [Bayesian Hybrid Matrix Factorisation for Data Integration"](https://github.com/ThomasBrouwer/HMF) project ([Thomas Brouwer and Pietro Lio', 2017](https://arxiv.org/abs/1704.04962)), using the complete datasets of each (before finding the overlap). For the GDSC IC50 dataset, some more details can be found in **/gdsc/notes**. 
- **/gdsc/load_data.py**, **/ccle/load_data.py** - Helper methods for loading in the Sanger, CCLE IC50, and CCLE EC50 datasets.

### /experiments/
- **/experiments_toy/** - Experiments on the toy data.
  - **/convergence/** - Measure convergence rate of the methods (against iterations) on the toy data.
  - **/time/** - Measure convergence rate of the methods (against time) on the toy data.
  - **/grid_search/** - Measure the effectiveness of the line, grid, and greedy search model selection methods on the toy data.
  - **/test_varying_missing/** - Measure the predictive performance on missing values for varying sparsity levels.
  - **/test_varying_noise/** - Measure the predictive performance on missing values for varying noise levels.
- **/experiments_gdsc/** - Experiments on the Sanger GDSC IC50 dataset, as well as helper methods for loading in the data.
  - **/convergence/** - Measure convergence rate of the methods (against iterations) on the Sanger data.
  - **/time/** - Measure convergence rate of the methods (against time) on the Sanger data.
  - **/grid_search/** - Measure the effectiveness of the line, grid, and greedy search model selection methods on the Sanger data.
  - **/cross_validation/** - 10-fold cross-validation experiments on the Sanger data.
- **/experiments_ccle/** - Cross-validation experiments on the CCLE IC50 and EC50 datasets, as well as helper methods for loading in the data.
  - **/cross_validation/** - 10-fold cross-validation experiments on the CCLE IC50 and EC50 data.

### /plots/
The results and plots for the experiments are stored in this folder, along with scripts for making the plots.
- **/graphs_toy/** - Plots for the experiments on the toy data.
- **/graphs_Sanger/** - Plots for the experiments on the Sanger GDSC drug sensitivity data.
- **/missing_values/** - Scripts for plotting the varying missing values experiment outcomes.
- **/model_selection/** - Scripts for plotting the model selection experiment outcomes.
- **/noise/** - Scripts for plotting the varying noise experiment outcomes.
- **/convergence/** - Scripts for plotting the convergence (against iterations) on the toy and Sanger data.
- **/time_toy/** - Scripts for plotting the convergence (against time) on the toy data.
- **/time_Sanger/** - Scripts for plotting the convergence (against time) on the Sanger data.

### /tests/
py.test unit tests for the code and classes in **/code/**. To run the tests, simply `cd` into the /tests/ folder, and run `pytest` in the command line.

</br>
</details>
