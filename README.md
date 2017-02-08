# BNMTF
Python implementations of bayesian non-negative matrix factorisation (BNMF) and tri-factorisation (BNMTF) algorithms, using Gibbs sampling and variational Bayesian inference. BNMF Gibbs sampler was introduced by Schmidt et al. (2009).

This project is structured as follows:

#### /code/
Implementations of the models.
- **/distributions/** - Contains code for obtaining draws of the exponential, Gaussian, and Truncated Normal distributions. Also has code for computing the expectation and variance of these distributions.
- **bnmf_gibbs.py** - Implementation of Gibbs sampler for Bayesian non-negative matrix factorisation (BNMF), extended to take into account missing values. Initially introduced by Schmidt et al. 2009.
- **bnmf_vb.py** - Implementation of our variational Bayesian inference for BNMF.
- **nmf_icm.py** - Implementation of Iterated Conditional Modes NMF algorithm (MAP inference). Initially introduced by Schmidt et al. 2009.
- **nmf_np.py** - Implementation of non-probabilistic NMF (Lee and Seung 2001).
- **bnmtf_gibbs.py ** - Implementation of our Gibbs sampler for Bayesian non-negative matrix tri-factorisation (BNMTF).
- **bnmtf_vb.py** - Implementation of our variational Bayesian inference for BNMTF.
- **nmtf_icm.py** - Implementation of Iterated Conditional Modes NMTF algorithm (MAP inference).
- **nmtf_np.py** - Implementation of non-probabilistic NMTF, introduced by Yoo and Choi 2009.

#### /tests/
py.test unit tests for the above mentioned code.

example/
	Contains code and data for trying the above code, with data generated from the model assumptions.

	generate_toy/
		Code for generating toy datasets from the model assumptions.

	recover_data/
		Code for recovering the latent matrices by using the BNMF and BNMTF models.
