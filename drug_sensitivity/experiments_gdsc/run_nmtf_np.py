"""
Run non-probabilistic NMF on the entire Sanger dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.nmtf_np import NMTF
from BNMTF.drug_sensitivity.experiments_gdsc.load_data import load_gdsc

import matplotlib.pyplot as plt

##########

standardised = False #standardised Sanger or unstandardised

iterations = 1000
I, J, K, L = 622,138,5, 5

init_S = 'exponential'
init_FG = 'kmeans'
expo_prior = 1/10.

# Load in data
(_,X_min,M,_,_,_,_) = load_gdsc(standardised=standardised)

# Run the algorithm
nmtf = NMTF(X_min,M,K,L) 
nmtf.initialise(init_S,init_FG,expo_prior)
nmtf.run(iterations)

# Print the performances across iterations (MSE)
print "all_performances = %s" % nmtf.all_performances['MSE']

# Plot the performances (MSE)
plt.plot(nmtf.all_performances['MSE'])