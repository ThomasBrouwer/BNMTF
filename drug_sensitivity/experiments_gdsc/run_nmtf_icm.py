"""
Run BNMTF with the Exp priors on the Sanger dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.nmtf_icm import nmtf_icm
from BNMTF.drug_sensitivity.experiments_gdsc.load_data import load_gdsc

import numpy, matplotlib.pyplot as plt, random

##########

standardised = False #standardised Sanger or unstandardised

iterations = 1000
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
NMTF = nmtf_icm(X_min,M,K,L,priors)
NMTF.initialise(init_S=init_S,init_FG=init_FG)
NMTF.run(iterations)

# Plot the tau expectation values to check convergence
plt.plot(NMTF.all_tau)

# Print the performances across iterations (MSE)
print "icm_all_performances = %s" % NMTF.all_performances['MSE']