"""
Run BNMTF with the Exp priors on the Sanger dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmtf_vb_optimised import bnmtf_vb_optimised
from BNMTF.drug_sensitivity.experiments_gdsc.load_data import load_gdsc

import numpy, matplotlib.pyplot as plt, random

##########

standardised = False #standardised Sanger or unstandardised

iterations = 1000
I, J, K, L = 622,138,5,5

init_S = 'random' #'exp' #
init_FG = 'kmeans' #'exp' #
tauFSG = {
    'tauF' : numpy.ones((I,K)),
    'tauS' : numpy.ones((K,L)),
    'tauG' : numpy.ones((J,L))  
}

alpha, beta = 1., 1.
lambdaF = numpy.ones((I,K))*1.
lambdaS = numpy.ones((K,L))/10.
lambdaG = numpy.ones((J,L))*1.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

# Load in data
(_,X_min,M,_,_,_,_) = load_gdsc(standardised=standardised)

#X_min = X_min[0:20,0:20]
#M = M[0:20,0:20]

# Run the Gibbs sampler
BNMTF = bnmtf_vb_optimised(X_min,M,K,L,priors)
BNMTF.initialise(init_S=init_S,init_FG=init_FG,tauFSG=tauFSG)
BNMTF.run(iterations)

# Plot the tau expectation values to check convergence
plt.plot(BNMTF.all_exp_tau)

# Print the performances across iterations (MSE)
print "vb_all_performances = %s" % BNMTF.all_performances['MSE']