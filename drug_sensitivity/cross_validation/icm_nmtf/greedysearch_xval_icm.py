"""
Run the cross validation with greedy search for model selection using ICM-NMTF 
on the Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.nmtf_icm import nmtf_icm
from BNMTF.cross_validation.greedy_search_cross_validation import GreedySearchCrossValidation
from BNMTF.drug_sensitivity.load_data import load_Sanger


# Settings
standardised = False
iterations = 1000

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

minimum_TN = 0.1

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised)

# Run the cross-validation framework
random.seed(42)
numpy.random.seed(9000)
nested_crossval = GreedySearchCrossValidation(
    classifier=nmtf_icm,
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
nested_crossval.run(minimum_TN=minimum_TN)

"""
All model fits for fold 1, metric AIC: [(4, 4, 259815.59627734334), (5, 4, 264878.72889689816), (4, 5, 259898.0246359052), (5, 5, 264190.02866958058)].
Best K,L for fold 1: (4, 4).
Performance: {'R^2': 0.7848296007416965, 'MSE': 2.500413159374383, 'Rp': 0.88598937222369789}.

All model fits for fold 2, metric AIC: [(4, 4, 260545.18275629985), (5, 4, 261275.96575320658), (4, 5, 260410.73482460013), (5, 5, 263173.61319014861), (4, 6, 260181.36536793038), (5, 6, 257747.22805994857), (6, 6, 258935.06227929285), (5, 7, 260389.90188614541), (6, 7, 257473.7647978157), (7, 7, 255711.64650259394), (6, 8, 256527.75340829749), (7, 8, 256018.57150287644), (8, 7, 256287.80302425858), (8, 8, 251931.91550177278)].
Best K,L for fold 2: (8, 8).
Performance: {'R^2': 0.8031655813109169, 'MSE': 2.2908378083534169, 'Rp': 0.89655601368355842}.

All model fits for fold 3, metric AIC: [(4, 4, 263869.44447429088), (5, 4, 264754.87779094913), (4, 5, 263525.47098255542), (5, 5, 263698.56896390981), (4, 6, 261261.34640233329), (5, 6, 260681.77037487802), (6, 6, 260926.80531871185), (5, 7, 260137.9268227336), (6, 7, 255578.55383834516), (7, 7, 255637.60252683982), (6, 8, 255150.27303801716), (7, 8, 255089.74074192555), (8, 8, 253033.99410260201)].
Best K,L for fold 3: (8, 8).
Performance: {'R^2': 0.7974086123797891, 'MSE': 2.4133750074853366, 'Rp': 0.89341450579824566}.

All model fits for fold 4, metric AIC: [(4, 4, 265233.25390790537), (5, 4, 264416.46376539336), (4, 5, 260394.20040612674), (5, 5, 258494.67912316788), (6, 5, 259310.22735996207), (5, 6, 260813.8947229542), (6, 6, 258600.4064698164)].
Best K,L for fold 4: (5, 5).
Performance: {'R^2': 0.7789817683950114, 'MSE': 2.5719353224488679, 'Rp': 0.88276201746274197}.

All model fits for fold 5, metric AIC: [(4, 4, 261293.85676520233), (5, 4, 262486.62483891868), (4, 5, 263015.61895033764), (5, 5, 261699.77768853033)].
Best K,L for fold 5: (4, 4).
Performance: {'R^2': 0.7737436658974457, 'MSE': 2.6395039288128728, 'Rp': 0.87969534617670053}.

all_MSE = [2.500413159374383,2.2908378083534169,2.4133750074853366,2.5719353224488679,2.6395039288128728]
all_R2 = [0.7848296007416965,0.8031655813109169,0.7974086123797891,0.7789817683950114,0.7737436658974457]

Average MSE: 2.483 +- 0.122
Average R^2: 0.788 +- 0.011
"""