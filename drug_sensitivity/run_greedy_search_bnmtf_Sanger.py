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

restarts = 1
iterations = 1000
I, J = 622,139
values_K = range(1,20+1)
values_L = range(1,20+1)

alpha, beta = 1., 1.
lambdaF = 1./10.
lambdaS = 1./10.
lambdaG = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

initFG = 'kmeans'
initS = 'random'

classifier = bnmtf_vb_optimised

search_metric = 'AIC'

# Load in data
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised)

folds_test = compute_folds(I,J,no_folds,M)
folds_training = compute_Ms(folds_test)
(M_train,M_test) = (folds_training[0],folds_test[0])

# Run the line search
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
greedy_search = GreedySearch(classifier,values_K,values_L,X_min,M,priors,initS,initFG,iterations,restarts=restarts)
greedy_search.search(search_metric)

# Plot the performances of all metrics
metrics = ['loglikelihood', 'BIC', 'AIC', 'MSE']
for metric in metrics:
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
    
    
# Also print out all values in a dictionary
all_values = {}
for metric in metrics:
    (_,_,values) = zip(*numpy.array(greedy_search.all_values(metric)))
    all_values[metric] = list(values)
    
print "all_values = %s \nlist_values_K=%s \nlist_values_L=%s" % \
    (all_values,list(list_values_K),list(list_values_L))

'''
all_values = {'MSE': [17.562631930477728, 17.562630540620951, 17.56262657828098, 3.972522040477652, 3.9720692998565821, 3.9720204651999724, 1.3284184587807626, 1.3266130480202567, 1.3286491775160303, 0.9369627101716429, 0.93286904568754703, 0.93372366696758535, 0.90233102956843292], 'loglikelihood': [-22816.119509697135, -22816.131024575923, -22816.129918299273, -16873.815443731917, -16873.42222510741, -16875.482561297165, -12497.670459344721, -12492.341988694485, -12501.539099323136, -11108.999771028019, -11096.732398576594, -11099.869729838247, -10968.639891130502], 'AIC': [45994.23901939427, 46196.262049151846, 46156.259836598547, 34475.630887463834, 34678.84445021482, 34642.965122594331, 26093.340918689442, 26288.68397738897, 26267.078198646272, 23689.999542056037, 23873.464797153189, 23839.739459676493, 23787.279782261005], 'BIC': [47258.921643934089, 48166.651552578522, 47986.905403611985, 37018.970530184793, 37934.878168643299, 37759.254904609574, 29929.311973232867, 30844.336304460576, 30682.986589304637, 28832.576402063249, 29742.710126509246, 29569.24085261931, 30250.436841373332]} 
list_values_K=[1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0] 
list_values_L=[1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0]
'''