This file contains some toy dataset examples.

/generate_toy/
	Scripts for generating toy datasets for NMF and BNMF, and the generated datasets themselves.

/recover_data/
	Script for running Gibbs or VB on the toy dataset, and analysing whether it works (i.e. Gibbs draws converge, VB error goes down with iterations).

/test_varying_missing/
	Test the performance of Gibbs and VB as the number of missing entries vary (but K is known). We use the toy dataset from /generate_toy/ but generate new mask matrices M.

/test_varying_error/
	Test the performance of Gibbs and VB as the error in generating the entries varies (but K is known). We use the toy dataset R_true.txt from /generate_toy/ but generate new noise.
