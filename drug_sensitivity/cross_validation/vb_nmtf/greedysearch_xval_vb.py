"""
Run the cross validation with greedy search for model selection using Gibbs-NMTF 
on the Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.bnmtf_vb_optimised import bnmtf_vb_optimised
from BNMTF.cross_validation.greedy_search_cross_validation import GreedySearchCrossValidation
from BNMTF.drug_sensitivity.load_data import load_Sanger


# Settings
standardised = False
iterations = 1000

init_S = 'random' #'exp' #
init_FG = 'kmeans' #'exp' #

K_range = [5,6,7,8]
L_range = [5,6,7,8]
no_folds = 5
restarts = 2

quality_metric = 'AIC'
output_file = "./results_TEST.txt"

alpha, beta = 1., 1.
lambdaF = 1./10.
lambdaS = 1./10.
lambdaG = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised)

# Run the cross-validation framework
random.seed(42)
numpy.random.seed(9000)
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
All model fits for fold 1, metric AIC: [(5, 5, 267436.21583785285), (6, 5, 268266.96019512357), (5, 6, 257633.99837328543), (6, 6, 258811.15017865735), (5, 7, 267685.94226889982), (6, 7, 259475.47788105684)].
Best K,L for fold 1: (5, 6).
Performance: {'R^2': 0.8054505750429645, 'MSE': 2.260784680365453, 'Rp': 0.89749750215470236}.

All model fits for fold 2, metric AIC: [(5, 5, 257397.3462916202), (6, 5, 258208.83672100294), (5, 6, 258507.40258252042), (6, 6, 268899.95363038778)].
Best K,L for fold 2: (5, 5).
Performance: {'R^2': 0.7932515856937303, 'MSE': 2.4062208604789466, 'Rp': 0.89070894807048406}.

All model fits for fold 3, metric AIC: [(5, 5, 257474.56778980303), (6, 5, 263139.31012906466), (5, 6, 257897.25841092237), (6, 6, 258977.56348133401)].
Best K,L for fold 3: (5, 5).
Performance: {'R^2': 0.795777258139038, 'MSE': 2.432808555965515, 'Rp': 0.89214912313681782}.

All model fits for fold 4, metric AIC: [(5, 5, 257590.35016277712), (6, 5, 257910.38864371926), (5, 6, 262678.73114371556), (6, 6, 268566.63052480557)].
Best K,L for fold 4: (5, 5).
Performance: {'R^2': 0.7904629976570847, 'MSE': 2.4383310542858903, 'Rp': 0.88915894594428779}.

All model fits for fold 5, metric AIC: [(5, 5, 257742.14887261484), (6, 5, 258285.20611216704), (5, 6, 257312.86952114664), (6, 6, 268671.71383575827), (5, 7, 267733.08592626522), (6, 7, 263382.4922671275)].
Best K,L for fold 5: (5, 6).
Performance: {'R^2': 0.7772741170646614, 'MSE': 2.5983177239567294, 'Rp': 0.88164537646013219}.

all_MSE = [2.260784680365453,2.4062208604789466,2.432808555965515,2.4383310542858903,2.5983177239567294]
all_R2 = [0.8054505750429645,0.7932515856937303,0.795777258139038,0.7904629976570847,0.7772741170646614]

Average MSE: 2.427 +- 0.107
Average R^2: 0.792 +- 0.009
"""