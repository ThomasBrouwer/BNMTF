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
All model fits for fold 1, metric AIC: [259448.77057675872, 255676.62051582715, 255523.47957909582, 254241.92682732039, 252679.74096095277].
Best K for fold 1: 10.
Performance: {'R^2': 0.8060286774889585, 'MSE': 2.2540667722869601, 'Rp': 0.89804470049443852}.

All model fits for fold 2, metric AIC: [257600.37492492507, 257018.22482604941, 255614.94666957104, 253635.59735767939, 253954.46612192781].
Best K for fold 2: 9.
Performance: {'R^2': 0.8055058118215439, 'MSE': 2.2636012682721058, 'Rp': 0.8978242230542316}.

All model fits for fold 3, metric AIC: [256887.89870918918, 257936.53215861766, 256188.7339930259, 255221.85440717341, 253671.36609325348].
Best K for fold 3: 10.
Performance: {'R^2': 0.8031286514017, 'MSE': 2.3452348985721208, 'Rp': 0.89656801110008155}.

All model fits for fold 4, metric AIC: [259802.08723315704, 257908.12596277907, 253929.13649055792, 254928.65295304605, 253336.49697324197].
Best K for fold 4: 10.
Performance: {'R^2': 0.7980388856579155, 'MSE': 2.3501722910618836, 'Rp': 0.89359502195937368}.

All model fits for fold 5, metric AIC: [259261.05530521285, 258289.01058679729, 255722.98660270439, 254669.80116246466, 253578.93580860901].
Best K for fold 5: 10.
Performance: {'R^2': 0.8046622351672897, 'MSE': 2.2788082365365399, 'Rp': 0.89723763908757848}.


all_MSE = [2.2540667722869601,2.2636012682721058,2.3452348985721208,2.3501722910618836,2.2788082365365399]
all_R2 = [0.8060286774889585,0.8055058118215439,0.8031286514017, 0.7980388856579155,0.8046622351672897]

Average MSE: 2.298 +- 0.041
Average R^2: 0.803 +- 0.003
"""