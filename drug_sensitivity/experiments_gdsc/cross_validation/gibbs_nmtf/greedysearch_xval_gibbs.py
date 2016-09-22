"""
Run the cross validation with greedy search for model selection using VB-NMTF 
on the Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.bnmtf_gibbs_optimised import bnmtf_gibbs_optimised
from BNMTF.cross_validation.greedy_search_cross_validation import GreedySearchCrossValidation
from BNMTF.drug_sensitivity.experiments_gdsc.load_data import load_gdsc


# Settings
standardised = False
iterations = 500
burn_in = 450
thinning = 2

init_S = 'random' #'exp' #
init_FG = 'kmeans' #'exp' #

K_range = [4,5,6,7,8]
L_range = [4,5,6,7,8]
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
random.seed(42)
numpy.random.seed(9000)
nested_crossval = GreedySearchCrossValidation(
    classifier=bnmtf_gibbs_optimised,
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
nested_crossval.run(burn_in=burn_in,thinning=thinning)

"""
All model fits for fold 1, metric AIC: [(4, 4, 262436.98303210537), (5, 4, 273407.76496789115), (4, 5, 263360.9277280402), (5, 5, 263400.5606913346)].
Best K,L for fold 1: (4, 4).
Performance: {'R^2': 0.7800291750013955, 'MSE': 2.5561970763677193, 'Rp': 0.88322370788808247}.

All model fits for fold 2, metric AIC: [(4, 4, 264745.89944073628), (5, 4, 267367.49610899633), (4, 5, 263978.79956275201), (5, 5, 265730.82631668856), (4, 6, 264306.77140626271), (5, 6, 266071.64877511811)].
Best K,L for fold 2: (4, 5).
Performance: {'R^2': 0.7793797705537095, 'MSE': 2.5676666015484755, 'Rp': 0.88294516494592834}.

All model fits for fold 3, metric AIC: [(4, 4, 269196.33589161438), (5, 4, 264955.89814824361), (4, 5, 266505.80081558786), (5, 5, 265883.96613683912), (6, 4, 270369.78494787007), (6, 5, 262221.11539569066), (7, 5, 268584.82367791957), (6, 6, 261034.21177468935), (7, 6, 265304.18808236072), (6, 7, 261208.71399229951), (7, 7, 262222.32652767492)].
Best K,L for fold 3: (6, 6).
Performance: {'R^2': 0.7852422451314671, 'MSE': 2.5583071637526698, 'Rp': 0.88627262096199066}.

All model fits for fold 4, metric AIC: [(4, 4, 264227.12581318244), (5, 4, 269245.63521912333), (4, 5, 267216.17738850694), (5, 5, 264999.82046250795)].
Best K,L for fold 4: (4, 4).
Performance: {'R^2': 0.7632666200639934, 'MSE': 2.7548086754594276, 'Rp': 0.87370141201889895}.

All model fits for fold 5, metric AIC: [(4, 4, 261662.41319516249), (5, 4, 270807.27778048464), (4, 5, 267437.31833433686), (5, 5, 264272.2503883256)].
Best K,L for fold 5: (4, 4).
Performance: {'R^2': 0.7657711026876297, 'MSE': 2.7325117643657997, 'Rp': 0.87515394367693233}.

all_MSE = [2.5561970763677193,2.5676666015484755,2.5583071637526698,2.7548086754594276,2.7325117643657997]
all_R2 = [0.7800291750013955,0.7793797705537095,0.7852422451314671,0.7632666200639934,0.7657711026876297]

Average MSE: 2.634 +- 0.089
Average R^2: 0.775 +- 0.009
"""