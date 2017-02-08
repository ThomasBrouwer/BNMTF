"""
Run the greedy grid search method for finding the best values for K and L for 
BNMTF. We use the parameters for the true priors.

The AIC seems to converge best to the true K,L (even when lambda = true_lambda/100).
"""


project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.data_toy.bnmtf.generate_bnmtf import generate_dataset, try_generate_M
from BNMTF.code.cross_validation.greedy_search_bnmtf import GreedySearch
from BNMTF.code.models.bnmtf_gibbs_optimised import bnmtf_gibbs_optimised

import numpy, matplotlib.pyplot as plt
import scipy.interpolate

##########

restarts = 5
iterations = 100
burn_in = 90
thinning = 2

I, J = 20,20
true_K, true_L = 3,3
values_K, values_L = range(1,5+1), range(1,5+1)

fraction_unknown = 0.1
attempts_M = 100

alpha, beta = 100., 1. #1., 1.
tau = alpha / beta
lambdaF = numpy.ones((I,true_K))
lambdaS = numpy.ones((true_K,true_L))
lambdaG = numpy.ones((J,true_L))

classifier = bnmtf_gibbs_optimised
initFG = 'kmeans'
initS = 'random'

search_metric = 'AIC'

# Generate data
(_,_,_,_,_,R) = generate_dataset(I,J,true_K,true_L,lambdaF,lambdaS,lambdaG,tau)
M = try_generate_M(I,J,fraction_unknown,attempts_M)

# Run the line search. The priors lambdaU and lambdaV need to be a single value (recall K is unknown)
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF[0,0], 'lambdaS':lambdaS[0,0], 'lambdaG':lambdaG[0,0] }
greedy_search = GreedySearch(classifier,values_K,values_L,R,M,priors,initS,initFG,iterations,restarts)
greedy_search.search(search_metric,burn_in,thinning)

# Plot the performances of all three metrics
for metric in ['loglikelihood', 'BIC', 'AIC', 'MSE']:
    # Make three lists of indices X,Y,Z (K,L,metric)
    KLvalues = numpy.array(greedy_search.all_values(metric))
    (list_values_K,list_values_L,values) = zip(*KLvalues)
    
    # Set up a regular grid of interpolation points
    Ki, Li = (numpy.linspace(min(list_values_K), max(list_values_K), 100), 
              numpy.linspace(min(list_values_L), max(list_values_L), 100))
    Ki, Li = numpy.meshgrid(Ki, Li)
    
    # Interpolate
    rbf = scipy.interpolate.Rbf(list_values_K, list_values_L, values, function='linear')
    values_i = rbf(Ki, Li)
    
    # Plot
    plt.figure()
    plt.imshow(values_i, cmap='jet_r', 
               vmin=min(values), vmax=max(values), origin='lower',
               extent=[min(list_values_K)-1, max(list_values_K)+1, min(list_values_L)-1, max(list_values_L)+1])
    plt.scatter(list_values_K, list_values_L, c=values, cmap='jet_r')
    plt.colorbar()
    plt.title("Metric: %s." % metric)   
    plt.xlabel("K")     
    plt.ylabel("L")  
    plt.show()
    
    # Print the best value
    best_K,best_L = greedy_search.best_value(metric)
    print "Best K,L for metric %s: %s,%s." % (metric,best_K,best_L)