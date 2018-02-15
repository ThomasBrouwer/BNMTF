"""
Run the line search method for finding the best value for K for BNMF.
We use the parameters for the true priors.

The BIC tends to give overly simple models, preferring K=1 oftentimes.
The log likelihood and AIC tend to peak at the true K if the correct priors are
given (this has to do with converging to a good local minimum).

If we give the wrong prior (true/5) we still obtain good convergence (with 
true*5 all values get pushed to 0, leading to terrible solutions), and we get
a nice peak for the log likelihood and AIC around the true K.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BNMTF.data_toy.bnmf.generate_bnmf import generate_dataset, try_generate_M
from BNMTF.code.cross_validation.line_search_bnmf import LineSearch
from BNMTF.code.models.bnmf_gibbs_optimised import bnmf_gibbs_optimised

import numpy, matplotlib.pyplot as plt

##########

restarts = 10
iterations = 1000
burn_in = 800
thinning = 5

I, J = 100, 80
true_K = 10
values_K = range(1,20+1)

fraction_unknown = 0.1
attempts_M = 100

alpha, beta = 1., 1. #1., 1.
tau = alpha / beta
lambdaU = numpy.ones((I,true_K))
lambdaV = numpy.ones((J,true_K))

classifier = bnmf_gibbs_optimised
initUV = 'random'

# Generate data
(_,_,_,_,R) = generate_dataset(I,J,true_K,lambdaU,lambdaV,tau)
M = numpy.ones((I,J))
#M = try_generate_M(I,J,fraction_unknown,attempts_M)

# Run the line search. The priors lambdaU and lambdaV need to be a single value (recall K is unknown)
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU[0,0]/10, 'lambdaV':lambdaV[0,0]/10 }
line_search = LineSearch(classifier,values_K,R,M,priors,initUV,iterations,restarts)
line_search.search(burn_in,thinning)

# Plot the performances of all three metrics - but MSE separately
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