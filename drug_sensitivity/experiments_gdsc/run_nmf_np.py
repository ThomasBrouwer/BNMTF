"""
Run non-probabilistic NMF on the entire Sanger dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.nmf_np import NMF
from BNMTF.drug_sensitivity.experiments_gdsc.load_data import load_Sanger

import matplotlib.pyplot as plt

##########

standardised = False #standardised Sanger or unstandardised

iterations = 10000
I, J, K = 622,138,10
init_UV = 'exponential'
expo_prior = 1/10.

# Load in data
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised)

# Run the algorithm
nmf = NMF(X_min,M,K) 
nmf.initialise(init_UV,expo_prior)
nmf.run(iterations)

# Print the performances across iterations (MSE)
print "all_performances = %s" % nmf.all_performances['MSE']

# Plot the performances (MSE)
plt.plot(nmf.all_performances['MSE'])