"""
Run the greedy grid search method for finding the best values for K and L for 
BNMTF. We use the parameters for the true priors.

The AIC seems to converge best to the true K,L (even when lambda = true_lambda/100).
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BNMTF.data_toy.bnmtf.generate_bnmtf import generate_dataset, try_generate_M
from BNMTF.code.cross_validation.greedy_search_bnmtf import GreedySearch
from BNMTF.code.models.bnmtf_vb_optimised import bnmtf_vb_optimised

import numpy, matplotlib.pyplot as plt
import scipy.interpolate

##########

restarts = 5
iterations = 1000

I, J = 100, 80
true_K, true_L = 5, 5
values_K, values_L = range(1,10+1), range(1,10+1)

#fraction_unknown = 0.1
attempts_M = 100

alpha, beta = 1., 1. #1., 1.
tau = alpha / beta
lambdaF = numpy.ones((I,true_K))
lambdaS = numpy.ones((true_K,true_L))
lambdaG = numpy.ones((J,true_L))

classifier = bnmtf_vb_optimised
initFG = 'kmeans'
initS = 'random'

search_metric = 'AIC'

# Generate data
(_,_,_,_,_,R) = generate_dataset(I,J,true_K,true_L,lambdaF,lambdaS,lambdaG,tau)
M  = numpy.ones((I,J))
#M = try_generate_M(I,J,fraction_unknown,attempts_M)

# Run the line search. The priors lambdaU and lambdaV need to be a single value (recall K is unknown)
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF[0,0]/10, 'lambdaS':lambdaS[0,0]/10, 'lambdaG':lambdaG[0,0]/10 }
greedy_search = GreedySearch(classifier,values_K,values_L,R,M,priors,initS,initFG,iterations,restarts)
greedy_search.search(search_metric)

# Plot the performances of all three metrics
metrics = ['loglikelihood', 'BIC', 'AIC', 'MSE', 'ELBO']
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

all_values_2 = {'ELBO': [-22359.943112635774, -22385.182965303662, -22384.928728440667, -20152.225795207723, -20120.081148784488, -20113.736904984518, -15718.206934168276, -15882.502246779264, -16082.733573069061, -14781.972227894446, -16445.685097824207, -16666.034522436297, -15248.392170040932], 'MSE': [12.297134409932569, 12.29713416492767, 12.297132457091566, 5.5793784869015646, 5.5831295525445768, 5.5815221190032318, 1.5138835709625185, 1.5092385737413609, 1.5134358399478318, 0.92962009847085136, 1.5104224758204408, 1.5129063690269584, 0.92624553633558482], 'loglikelihood': [-21390.491317875087, -21390.503065023542, -21390.502967566365, -18232.531033907821, -18235.267019764935, -18234.096911606917, -13020.29118621323, -13008.276977135356, -13022.246533559051, -11077.523760384123, -13017.693154745783, -13024.887236684059, -11067.410356479686], 'AIC': [43142.982635750173, 43345.006130047084, 43305.005935132729, 37193.062067815641, 37402.53403952987, 37360.193823213835, 27138.582372426459, 27320.553954270712, 27308.493067118103, 23627.047520768247, 27715.386309491565, 27689.774473368117, 23984.820712959372], 'BIC': [44407.665260289992, 45315.39563347376, 45135.651502146167, 39736.4017105366, 40658.56775795835, 40476.483605229078, 30974.553426969884, 31876.206281342318, 31724.401457776468, 28769.624380775458, 33584.631638847626, 33419.275866310934, 30447.977772071696]} 
list_values_K_2=[1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0] 
list_values_L_2=[1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0]
'''