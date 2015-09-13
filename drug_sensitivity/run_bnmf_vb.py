"""
Run BNMTF with the Exp priors on the Sanger dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmf_vb import bnmf_vb
from ml_helpers.code.mask import compute_Ms, compute_folds
from load_data import load_Sanger

import numpy, matplotlib.pyplot as plt

##########

standardised = False #standardised Sanger or unstandardised
no_folds = 5

iterations = 1000
init = 'random'
I, J, K = 622,139,10

alpha, beta = 1., 1.
lambdaU = numpy.ones((I,K))
lambdaV = numpy.ones((J,K))
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

# Load in data
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised)

folds_test = compute_folds(I,J,no_folds,M)
folds_training = compute_Ms(folds_test)
(M_train,M_test) = (folds_training[0],folds_test[0])

# Run the Gibbs sampler
BNMF = bnmf_vb(X_min,M,K,priors)
BNMF.initialise()
BNMF.run(iterations)

# Also measure the performances
performances = BNMF.predict(M_test)
print performances

# Plot the tau expectation values to check convergence
plt.plot(BNMF.all_exp_tau)