"""
Run the greedy grid search for BNMTF with the Exp priors on the Sanger dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmtf_vb_optimised import bnmtf_vb_optimised
from ml_helpers.code.mask import compute_Ms, compute_folds
from load_data import load_Sanger
from BNMTF.grid_search.greedy_search_bnmtf import GreedySearch

import numpy, matplotlib.pyplot as plt
import scipy.interpolate

##########

standardised = False #standardised Sanger or unstandardised
no_folds = 5

iterations = 1000
I, J = 622,139
values_K = range(1,10+1)
values_L = range(1,10+1)

alpha, beta = 1., 1.
lambdaF = 1
lambdaS = 1
lambdaG = 1
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

initFG = 'kmeans'
initS = 'random'

search_metric = 'AIC'

# Load in data
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised)

folds_test = compute_folds(I,J,no_folds,M)
folds_training = compute_Ms(folds_test)
(M_train,M_test) = (folds_training[0],folds_test[0])

# Run the line search
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
greedy_search = GreedySearch(values_K,values_L,X_min,M,priors,initS,initFG,iterations)
greedy_search.search(search_metric)

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
    plt.imshow(values_i, vmin=min(values), vmax=max(values), origin='lower',
           extent=[min(list_values_K)-1, max(list_values_K)+1, min(list_values_L)-1, max(list_values_L)+1])
    plt.scatter(list_values_K, list_values_L, c=values)
    plt.colorbar()
    plt.title("Metric: %s." % metric)   
    plt.xlabel("K")     
    plt.ylabel("L")  
    plt.show()
    
    # Print the best value
    best_K,best_L = greedy_search.best_value(metric)
    print "Best K,L for metric %s: %s,%s." % (metric,best_K,best_L)