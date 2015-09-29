"""
Run the grid search for BNMTF with the Exp priors on the Sanger dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmtf_vb_optimised import bnmtf_vb_optimised
from ml_helpers.code.mask import compute_Ms, compute_folds
from load_data import load_Sanger
from BNMTF.grid_search.grid_search_bnmtf import GridSearch

import numpy, matplotlib.pyplot as plt
import scipy.interpolate

##########

standardised = False #standardised Sanger or unstandardised
no_folds = 5

restarts = 1
iterations = 10
I, J = 622,139
values_K = range(1,2+1)
values_L = range(1,2+1)

alpha, beta = 1., 1.
lambdaF = 1./10.
lambdaS = 1./10.
lambdaG = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

initFG = 'kmeans'
initS = 'random'

classifier = bnmtf_vb_optimised

# Load in data
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised)

folds_test = compute_folds(I,J,no_folds,M)
folds_training = compute_Ms(folds_test)
(M_train,M_test) = (folds_training[0],folds_test[0])

# Run the line search
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
grid_search = GridSearch(classifier,values_K,values_L,X_min,M,priors,initS,initFG,iterations,restarts=restarts)
grid_search.search()

# Plot the performances of all metrics
metrics = ['loglikelihood', 'BIC', 'AIC','MSE']
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
    rbf = scipy.interpolate.Rbf(list_values_K, list_values_L, values, function='linear')
    values_i = rbf(Ki, Li)
    
    # Plot
    plt.figure()
    plt.imshow(values_i, cmap='jet_r',
               vmin=min(values), vmax=max(values), origin='lower',
               extent=[min(values_K), max(values_K), min(values_L), max(values_L)])
    plt.scatter(list_values_K, list_values_L, c=values, cmap='jet_r')
    plt.colorbar()
    plt.title("Metric: %s." % metric)   
    plt.xlabel("K")     
    plt.ylabel("L")  
    plt.show()
    
    # Print the best value
    best_K,best_L = grid_search.best_value(metric)
    print "Best K,L for metric %s: %s,%s." % (metric,best_K,best_L)
    
    
# Also print out all values in a dictionary
all_values = {}
for metric in metrics:
    all_values[metric] = list(grid_search.all_values(metric).flatten())
    
print "all_values = %s \nvalues_K=%s \nvalues_L=%s" % (all_values,values_K,values_L)

'''

'''