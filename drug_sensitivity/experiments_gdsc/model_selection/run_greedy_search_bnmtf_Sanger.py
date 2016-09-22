"""
Run the greedy grid search for BNMTF with the Exp priors on the Sanger dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmtf_vb_optimised import bnmtf_vb_optimised
from ml_helpers.code.mask import compute_Ms, compute_folds
from BNMTF.drug_sensitivity.experiments_gdsc.load_data import load_gdsc
from BNMTF.grid_search.greedy_search_bnmtf import GreedySearch

import numpy, matplotlib.pyplot as plt
import scipy.interpolate

##########

standardised = False #standardised Sanger or unstandardised
no_folds = 5

restarts = 5
iterations = 1000
I, J = 622,138
values_K = range(1,30+1)
values_L = range(1,30+1)

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
(_,X_min,M,_,_,_,_) = load_gdsc(standardised=standardised)

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
all_values = {'MSE': [3.0272042551947203, 3.027204256305112, 3.0272042923576148, 2.5914654932112464, 2.5918836849320201, 2.5914602381010914, 2.3493739958858635, 2.3511225674996381, 2.3584324978814539, 2.1868222893761833, 2.1911559705091568, 2.2016668628098452, 2.0510257720785683, 2.0546897432717603, 2.0586496735360251, 2.0826309185454925], 'loglikelihood': [-138379.73430838491, -138379.74014614287, -138380.57362950334, -132935.31284836732, -132949.35927074254, -132936.87960196467, -129506.33023969264, -129543.6258747291, -129641.90589727147, -127005.00858615834, -127072.80234078577, -127243.67779180428, -124768.25065830135, -124835.58530247367, -124903.988439383, -125310.14633691005], 'AIC': [278283.46861676982, 279529.48029228573, 278565.14725900668, 268922.62569673464, 270198.71854148508, 269207.75920392934, 263596.66047938529, 264921.2517494582, 264151.81179454294, 260130.01717231667, 261517.60468157154, 260893.35558360856, 257196.50131660269, 258585.17060494734, 257755.976878766, 259824.29267382011], 'BIC': [285262.09744653094, 292213.73348023742, 286825.93886588927, 282898.19996735867, 289889.07547585049, 284474.6545572257, 284587.4968019739, 291636.02904133906, 286443.12750535476, 288154.43215797161, 295275.11894206965, 290227.40826303756, 292272.81157642574, 299403.7384451644, 294151.08313791396, 301970.81481891294]} 
list_values_K=[1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0] 
list_values_L=[1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0]
'''