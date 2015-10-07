"""
Run the cross validation with line search for model selection using VB-NMF on
the Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.bnmf_gibbs_optimised import bnmf_gibbs_optimised
from BNMTF.cross_validation.line_search_cross_validation import LineSearchCrossValidation
from BNMTF.drug_sensitivity.load_data import load_Sanger


# Settings
standardised = False
iterations = 500#200
burn_in = 450#180
thinning = 2
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
    classifier=bnmf_gibbs_optimised,
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
nested_crossval.run(burn_in=burn_in,thinning=thinning)

"""
All model fits for fold 1, metric AIC: [259448.77057675872, 258235.11879871984, 255573.58954482598, 254696.85024825184, 253206.64588975901].
Best K for fold 1: 10.
Performance: {'R^2': 0.8038426722870428, 'MSE': 2.2794695051543621, 'Rp': 0.8968753770114728}.

All model fits for fold 2, metric AIC: [260400.24337271467, 257624.1975411509, 254755.12547332968, 255228.44332235208, 254072.04815624456].
Best K for fold 2: 10.
Performance: {'R^2': 0.8022773607196355, 'MSE': 2.3011752753788319, 'Rp': 0.89620339537704274}.

All model fits for fold 3, metric AIC: [258929.39611877583, 258192.1983796563, 255345.36256242247, 255920.19707536796, 253437.05414550015].
Best K for fold 3: 10.
Performance: {'R^2': 0.8036644470434058, 'MSE': 2.3388522194957702, 'Rp': 0.8968181876867628}.

All model fits for fold 4, metric AIC: [258670.68591149745, 259269.74693149255, 254618.2287971254, 254864.55360125337, 254099.04658395957].
Best K for fold 4: 10.
Performance: {'R^2': 0.7992844705144458, 'MSE': 2.3356777235035682, 'Rp': 0.89434039208251104}.

All model fits for fold 5, metric AIC: [258343.1164455777, 257839.49106167446, 255035.05525613483, 253573.65695192612, 253520.87549212185].
Best K for fold 5: 10.
Performance: {'R^2': 0.8047119529751179, 'MSE': 2.2782282291320404, 'Rp': 0.89729843931779008}.

all_MSE = [2.2794695051543621,2.3011752753788319,2.3388522194957702,2.3356777235035682,2.2782282291320404]
all_R2 = [0.8038426722870428,0.8022773607196355,0.8036644470434058, 0.7992844705144458,0.8047119529751179]

Average MSE: 2.306 +- 0.026
Average R^2: 0.803 +- 0.002
"""