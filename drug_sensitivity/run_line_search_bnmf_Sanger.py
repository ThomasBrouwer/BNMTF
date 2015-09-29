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
all_values = {'MSE': [3.0272045911768402, 2.5912786963651278, 2.3429500575625224, 2.1579551300728168, 2.0087568017872255, 1.8833097310478804, 1.7880018679366252, 1.710385377331598, 1.6419202215045337, 1.5731983786734194, 1.5114966663007683, 1.4557785810400978, 1.4133904459699074, 1.363169871115367, 1.3138117050399427, 1.2735819600049385, 1.2317154407491382, 1.2051820250897016, 1.1517577159221375, 1.1277304779924728], 'loglikelihood': [-138379.73269637415, -132932.74194889492, -129410.17189680642, -126540.47674235499, -124047.10357170555, -121807.48595337186, -120013.51345227598, -118486.68394554731, -117090.25218129493, -115630.28114463923, -114273.03960506352, -113007.81582288713, -112015.61679335347, -110808.59251800993, -109573.08840862897, -108548.71406382117, -107457.74917927168, -106751.28778736616, -105269.09185795794, -104600.34998176023], 'AIC': [278281.4653927483, 268909.48389778985, 263386.34379361285, 259168.95348470999, 255704.2071434111, 252746.97190674371, 250681.02690455195, 249149.36789109462, 247878.50436258985, 246480.56228927846, 245288.07921012703, 244279.63164577427, 243817.23358670695, 242925.18503601986, 241976.17681725795, 241449.42812764234, 240789.49835854335, 240898.57557473233, 239456.18371591589, 239640.69996352046], 'BIC': [285250.93591695855, 282848.42494621041, 284294.75536624365, 287046.83558155107, 290551.55976446246, 294563.79505200533, 299467.32057402382, 304905.13208477676, 310603.73908048228, 316175.26753138116, 321952.25497643999, 327913.27793629747, 334420.35040144046, 340497.77237496362, 346518.23468041199, 352960.9565150066, 359270.49727011792, 366349.04501051712, 371876.12367591099, 379030.11044772586]}
'''