"""
Run the cross validation with greedy search for model selection using Gibbs-NMTF 
on the Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.bnmtf_vb_optimised import bnmtf_vb_optimised
from BNMTF.cross_validation.greedy_search_cross_validation import GreedySearchCrossValidation
from BNMTF.drug_sensitivity.experiments_gdsc.load_data import load_gdsc


# Settings
standardised = False
iterations = 1000

init_S = 'random' #'exp' #
init_FG = 'kmeans' #'exp' #

K_range = [5,6,7,8,9,10]
L_range = [5,6,7,8,9,10]
no_folds = 10
restarts = 1

quality_metric = 'AIC'
output_file = "./REDO_results.txt"

alpha, beta = 1., 1.
lambdaF = 1./10.
lambdaS = 1./10.
lambdaG = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_gdsc(standardised=standardised)

# Run the cross-validation framework
#random.seed(42)
#numpy.random.seed(9000)
nested_crossval = GreedySearchCrossValidation(
    classifier=bnmtf_vb_optimised,
    R=X_min,
    M=M,
    values_K=K_range,
    values_L=L_range,
    folds=no_folds,
    priors=priors,
    init_S=init_S,
    init_FG=init_FG,
    iterations=iterations,
    restarts=restarts,
    quality_metric=quality_metric,
    file_performance=output_file
)
nested_crossval.run()

"""
all_MSE = [2.2811777476249415, 2.1782935772707153, 2.3760214934948851, 2.4070138866182651, 2.1679193763392863, 2.4351661211853344, 2.3531667160686407, 2.4375820084579578, 2.1737221434522502, 2.3957602752026799]
all_R2 = [0.8004514561880776, 0.8095655871226215, 0.7982332012844026, 0.7939011733335062, 0.8135460410954071, 0.7914028391107459, 0.8050979272119902, 0.7964032435159856, 0.8102340265362746, 0.805071751458151]

Average MSE: 2.3205823345714953 +- 0.011074845252916733
Average R^2: 0.80239072468571615 +- 5.0165577464731684e-05
"""