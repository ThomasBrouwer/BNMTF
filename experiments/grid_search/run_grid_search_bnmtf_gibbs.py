"""
Run the grid search method for finding the best values for K and L for BNMTF.
We use the parameters for the true priors.

For BNMTF I find that the BIC is a better estimator - the log likelihood is 
high for higher values for K and L than the true ones, same for the AIC. With
the BIC we get a nice peak just below the true K and L (for true K=L=5, at K=L=4).
"""


project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.experiments.generate_toy.bnmtf.generate_bnmtf import generate_dataset, try_generate_M
from BNMTF.grid_search.grid_search_bnmtf import GridSearch
from BNMTF.code.bnmtf_gibbs_optimised import bnmtf_gibbs_optimised

import numpy, matplotlib.pyplot as plt
import scipy.interpolate

##########

iterations = 100
burn_in = 90
thinning = 2

I, J = 20,20
true_K, true_L = 2,2
values_K, values_L = range(1,4+1), range(1,4+1)

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

# Generate data
(_,_,_,_,_,R) = generate_dataset(I,J,true_K,true_L,lambdaF,lambdaS,lambdaG,tau)
M = try_generate_M(I,J,fraction_unknown,attempts_M)

# Run the line search. The priors lambdaF,S,G need to be a single value (recall K,L is unknown)
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF[0,0], 'lambdaS':lambdaS[0,0], 'lambdaG':lambdaG[0,0] }
grid_search = GridSearch(classifier,values_K,values_L,R,M,priors,initS,initFG,iterations)
grid_search.search(burn_in,thinning)

# Plot the performances of all three metrics
for metric in ['loglikelihood', 'BIC', 'AIC','MSE']:
    # Make three lists of indices X,Y,Z (K,L,metric)
    values = numpy.array(grid_search.all_values(metric)).flatten()
    list_values_K = numpy.array([values_K for l in range(0,len(values_L))]).T.flatten()
    list_values_L = numpy.array([values_L for k in range(0,len(values_K))]).flatten()
    
    # Set up a regular grid of interpolation points
    Ki, Li = (numpy.linspace(min(list_values_K), max(list_values_K), 100), 
              numpy.linspace(min(list_values_L), max(list_values_L), 100))
    Ki, Li = numpy.meshgrid(Ki, Li)
    
    # Interpolate
    rbf = scipy.interpolate.Rbf(list_values_K, list_values_L, values, function='multiquadric')
    values_i = rbf(Ki, Li)
    
    # Plot
    plt.figure()
    plt.imshow(values_i, vmin=min(values), vmax=max(values), origin='lower',
           extent=[min(values_K), max(values_K), min(values_L), max(values_L)])
    plt.scatter(list_values_K, list_values_L, c=values)
    plt.colorbar()
    plt.title("Metric: %s." % metric)   
    plt.xlabel("K")     
    plt.ylabel("L")  
    plt.show()
    
    # Print the best value
    best_K,best_L = grid_search.best_value(metric)
    print "Best K,L for metric %s: %s,%s." % (metric,best_K,best_L)