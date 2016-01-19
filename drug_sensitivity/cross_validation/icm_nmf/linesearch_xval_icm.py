"""
Run the cross validation with line search for model selection using VB-NMF on
the Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.nmf_icm import nmf_icm
from BNMTF.cross_validation.line_search_cross_validation import LineSearchCrossValidation
from BNMTF.drug_sensitivity.load_data import load_Sanger


# Settings
standardised = False
iterations = 200
init_UV = 'random'

K_range = [6,7,8,9,10]#[1,5,10,15,20,25,30]
no_folds = 5
restarts = 2

quality_metric = 'AIC'
output_file = "./results_TEST.txt"

alpha, beta = 1., 1.
lambdaU = 1./10.
lambdaV = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

minimum_TN = 0.1

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised,sep=',')

# Run the cross-validation framework
random.seed(42)
numpy.random.seed(9000)
nested_crossval = LineSearchCrossValidation(
    classifier=nmf_icm,
    R=X_min,
    M=M,
    values_K=K_range,
    folds=no_folds,
    priors=priors,
    init_UV=init_UV,
    iterations=iterations,
    restarts=restarts,
    quality_metric=quality_metric,
    file_performance=output_file
)
nested_crossval.run(minimum_TN=minimum_TN)

"""
All model fits for fold 1, metric AIC: [252997.91066856537, 250747.87405923707, 249171.33422154275, 248348.40470946211, 246802.03487854672].
Best K for fold 1: 10.
Performance: {'R^2': 0.807748962293107, 'MSE': 2.2340759985699807, 'Rp': 0.89958671281823332}.

All model fits for fold 2, metric AIC: [252993.4545215781, 250747.87405923707, 249171.33422154275, 248348.40470946211, 246802.03487854672].
Best K for fold 2: 10.
Performance: {'R^2': 0.8075286864183611, 'MSE': 2.2400582434352443, 'Rp': 0.89953229833400639}.

All model fits for fold 3, metric AIC: [252998.07360170182, 250747.87405923707, 249171.33422154275, 248348.40470946211, 246802.03487854672].
Best K for fold 3: 10.
Performance: {'R^2': 0.8043624727400118, 'MSE': 2.3305369708045012, 'Rp': 0.89777844572303478}.

All model fits for fold 4, metric AIC: [252998.33798114926, 250747.87405923707, 249171.33422154275, 248348.40470946211, 246802.03487854672].
Best K for fold 4: 10.
Performance: {'R^2': 0.8020475315100168, 'MSE': 2.3035246557634794, 'Rp': 0.89627310905893576}.

All model fits for fold 5, metric AIC: [252998.18879330286, 250747.87405923707, 249171.33422154275, 248348.40470946211, 246802.03487854672].
Best K for fold 5: 10.
Performance: {'R^2': 0.8080801550156258, 'MSE': 2.2389348208205333, 'Rp': 0.89962819094414426}.

all_MSE = [2.2340759985699807,2.2400582434352443,2.3305369708045012,2.3035246557634794,2.2389348208205333]
all_R2 = [0.807748962293107,0.8075286864183611,0.8043624727400118,0.8020475315100168,0.8080801550156258]

Average MSE: 2.269 +- 0.040
Average R^2: 0.806 +- 0.002
"""