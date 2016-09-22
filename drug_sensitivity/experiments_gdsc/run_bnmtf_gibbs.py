"""
Run BNMTF with the Exp priors on the Sanger dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmtf_gibbs_optimised import bnmtf_gibbs_optimised
from BNMTF.drug_sensitivity.experiments_gdsc.load_data import load_gdsc

import numpy, matplotlib.pyplot as plt, random

##########

standardised = False #standardised Sanger or unstandardised

iterations = 1000
burnin = 800
thinning = 5

I, J, K, L = 622,138,5,5
init_S = 'random' #'exp' #
init_FG = 'kmeans' #'exp' #

alpha, beta = 1., 1.
lambdaF = numpy.ones((I,K))/10.
lambdaS = numpy.ones((K,L))/10.
lambdaG = numpy.ones((J,L))/10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

# Load in data
(_,X_min,M,_,_,_,_) = load_gdsc(standardised=standardised)

# Run the Gibbs sampler
BNMTF = bnmtf_gibbs_optimised(X_min,M,K,L,priors)
BNMTF.initialise(init_S=init_S,init_FG=init_FG)
BNMTF.run(iterations)

# Also measure the performances on the training data
performances = BNMTF.predict(M,burnin,thinning)
print performances

# Plot the tau expectation values to check convergence
plt.plot(BNMTF.all_tau)

# Print the performances across iterations (MSE)
print "all_performances = %s" % BNMTF.all_performances['MSE']
