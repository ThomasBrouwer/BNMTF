"""
Run the cross validation with line search for model selection using VB-NMF on
the Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.bnmf_vb_optimised import bnmf_vb_optimised
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

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised)

# Run the cross-validation framework
random.seed(42)
numpy.random.seed(9000)
nested_crossval = LineSearchCrossValidation(
    classifier=bnmf_vb_optimised,
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
nested_crossval.run()

"""
All model fits for fold 1, metric AIC: [255811.96901730512, 252136.60278205923, 250856.47342754889, 250226.77983334119, 248494.30974350378].
Best K for fold 1: 10.
Performance: {'R^2': 0.8062985415424776, 'MSE': 2.2509307850276188, 'Rp': 0.89865828998196062}.

All model fits for fold 2, metric AIC: [254648.23575785581, 252738.04961239398, 251251.00158825511, 249894.09518978855, 249008.72134400008].
Best K for fold 2: 10.
Performance: {'R^2': 0.8034479975799519, 'MSE': 2.2875509346902207, 'Rp': 0.8971303533111219}.

All model fits for fold 3, metric AIC: [254815.81149370887, 252028.20607615853, 251536.61512286673, 249416.79922148222, 249128.1772425485].
Best K for fold 3: 10.
Performance: {'R^2': 0.8053582521009495, 'MSE': 2.3186747240876673, 'Rp': 0.89811845194979079}.

All model fits for fold 4, metric AIC: [253723.76692031277, 253550.89846148193, 251558.85284609973, 249711.78828414652, 249128.15196378558].
Best K for fold 4: 10.
Performance: {'R^2': 0.8017121128828982, 'MSE': 2.3074278406212581, 'Rp': 0.89594935537176135}.

All model fits for fold 5, metric AIC: [254496.13874081109, 251849.87041349761, 250987.09739078008, 250049.4800796271, 248542.51724879385].
Best K for fold 5: 10.
Performance: {'R^2': 0.8063430635650138, 'MSE': 2.2591997107595674, 'Rp': 0.89846044831157224}.

all_MSE = [2.2509307850276188,2.2875509346902207,2.3186747240876673,2.3074278406212581,2.2591997107595674]
all_R2 = [0.8062985415424776,0.8034479975799519,0.8053582521009495,0.8017121128828982,0.8063430635650138]

Average MSE: 2.285 +- 0.026
Average R^2: 0.805 +- 0.002
"""