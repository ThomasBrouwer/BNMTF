"""
Run the cross validation with greedy search for model selection using VB-NMTF 
on the Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.bnmtf_gibbs_optimised import bnmtf_gibbs_optimised
from BNMTF.cross_validation.greedy_search_cross_validation import GreedySearchCrossValidation
from BNMTF.drug_sensitivity.load_data import load_Sanger


# Settings
standardised = False
iterations = 1000#2000
burn_in = 950#1950
thinning = 2

init_S = 'random' #'exp' #
init_FG = 'kmeans' #'exp' #

K_range = [4,5,6,7,8]
L_range = [4,5,6,7,8]
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
All model fits for fold 1, metric AIC: [(4, 4, 262436.98303210537), (5, 4, 269910.09760611271), (4, 5, 265197.495566628), (5, 5, 263798.09644951375)].
Best K,L for fold 1: (4, 4).
Performance: {'R^2': 0.767107132786292, 'MSE': 2.7063592014184272, 'Rp': 0.87587535847162934}.

All model fits for fold 2, metric AIC: [(4, 4, 267873.32948196388), (5, 4, 271326.15735661192), (4, 5, 268403.35639991832), (5, 5, 268821.65091122693)].
Best K,L for fold 2: (4, 4).
Performance: {'R^2': 0.7691938049601201, 'MSE': 2.6862149491992007, 'Rp': 0.87706696654070126}.

All model fits for fold 3, metric AIC: [(4, 4, 268664.5944636811), (5, 4, 273431.74266691034), (4, 5, 264508.61850172153), (5, 5, 267257.79733246739), (4, 6, 262426.11820365326), (5, 6, 264652.01927647821), (4, 7, 268114.2324529296), (5, 7, 264649.85086586827)].
Best K,L for fold 3: (4, 6).
Performance: {'R^2': 0.7733665036344839, 'MSE': 2.6997772334374921, 'Rp': 0.87946292253984737}.

All model fits for fold 4, metric AIC: [(4, 4, 270863.94686130306), (5, 4, 268407.91794441664), (4, 5, 264401.31276846648), (5, 5, 264476.35152452579), (4, 6, 265124.69181951357), (5, 6, 263803.07149369916), (6, 6, 264666.90513682854), (5, 7, 261132.43660203085), (6, 7, 263052.8809111907), (5, 8, 264169.26039988303), (6, 8, 263135.5321983592)].
Best K,L for fold 4: (5, 7).
Performance: {'R^2': 0.7776856994811916, 'MSE': 2.5870173606842415, 'Rp': 0.88200709307986647}.

All model fits for fold 5, metric AIC: [(4, 4, 263787.15012093011), (5, 4, 268664.40412508231), (4, 5, 263132.63955062849), (5, 5, 268883.9891101754), (4, 6, 264526.22998336912), (5, 6, 263758.00481077819)].
Best K,L for fold 5: (4, 5).
Performance: {'R^2': 0.7722883536322204, 'MSE': 2.6564815858448774, 'Rp': 0.8788522437150379}.

all_MSE = [2.7063592014184272,2.6862149491992007,2.6997772334374921,2.5870173606842415,2.6564815858448774]
all_R2 = [0.767107132786292,0.7691938049601201,0.7733665036344839,0.7776856994811916,0.7722883536322204]

Average MSE: 2.667 +- 0.044
Average R^2: 0.772 +- 0.004
"""