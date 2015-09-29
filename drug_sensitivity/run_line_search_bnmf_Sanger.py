"""
Run the line search for BNMF with the Exp priors on the Sanger dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmf_vb_optimised import bnmf_vb_optimised
from ml_helpers.code.mask import compute_Ms, compute_folds
from load_data import load_Sanger
from BNMTF.grid_search.line_search_bnmf import LineSearch

import numpy, matplotlib.pyplot as plt

##########

standardised = False #standardised Sanger or unstandardised
no_folds = 5

restarts = 1
iterations = 1000
I, J = 622,139
values_K = range(1,20+1)

alpha, beta = 1., 1.
lambdaU = 1./10.
lambdaV = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

initUV = 'random'

classifier = bnmf_vb_optimised

# Load in data
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised)

folds_test = compute_folds(I,J,no_folds,M)
folds_training = compute_Ms(folds_test)
(M_train,M_test) = (folds_training[0],folds_test[0])

# Run the line search
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
line_search = LineSearch(classifier,values_K,X_min,M,priors,initUV,iterations,restarts=restarts)
line_search.search()

# Plot the performances of all four metrics
metrics = ['loglikelihood', 'BIC', 'AIC', 'MSE']
for metric in metrics:
    plt.figure()
    plt.plot(values_K, line_search.all_values(metric), label=metric)
    plt.legend(loc=3)
    
# Also print out all values in a dictionary
all_values = {}
for metric in metrics:
    all_values[metric] = line_search.all_values(metric)
    
print "all_values = %s" % all_values

'''

'''