# BNMTF
Python implementations of bayesian non-negative matrix factorisation (BNMF) and tri-factorisation (BNMTF) algorithms, using Gibbs sampling and variational Bayesian inference. BNMF Gibbs sampler was introduced by Schmidt et al. (2009).

This project is structured as follows:

/code
	/distributions
		Contains code for obtaining draws of the exponential, Gaussian, and Truncated Normal distributions. Also has code for computing the expectation and variance of these distributions.

	/bnmf_gibbs
		Implementation of Gibbs sampler introduced by Schmidt et al. for Bayesian non-negative matrix factorisation (BNMF), extended to take into account missing values.

	/bnmf_vb
		Implementation of my own variational Bayesian inference for BNMF.

	/bnmtf_gibbs
		Implementation of Gibbs sampler for Bayesian non-negative matrix tri-factorisation (BNMTF).

	/bnmtf_vb
		Implementation of my own variational Bayesian inference for BNMTF.

/tests
	py.test unit tests for the above mentioned code.

/example
	Contains code and data for trying the above code, with data generated from the model assumptions.

	/generate_toy
		Code for generating toy datasets from the model assumptions.

	/recover_data
		Code for recovering the latent matrices by using the BNMF and BNMTF models.
