"""
Run NMF Iterated Conditional Modes with the Exp priors on the entire Sanger dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.nmf_icm import nmf_icm
from BNMTF.drug_sensitivity.experiments_gdsc.load_data import load_Sanger

import numpy, matplotlib.pyplot as plt

##########

standardised = False #standardised Sanger or unstandardised

iterations = 1000
init_UV = 'random'
I, J, K = 622,138,10

alpha, beta = 1., 1.
lambdaU = numpy.ones((I,K))/10.
lambdaV = numpy.ones((J,K))/10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

# Load in data
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised)

# Run the algorithm
NMF = nmf_icm(X_min,M,K,priors)
NMF.initialise(init_UV)
NMF.run(iterations)

# Plot the tau expectation values to check convergence
plt.plot(NMF.all_tau)

# Print the performances across iterations (MSE)
print "all_performances = %s" % NMF.all_performances['MSE']
