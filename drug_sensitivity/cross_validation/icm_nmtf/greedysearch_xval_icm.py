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
output_file = "./results.txt"

alpha, beta = 1., 1.
lambdaF = 1./10.
lambdaS = 1./10.
lambdaG = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

minimum_TN = 0.1

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised,sep=',')

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

All model fits for fold 2, metric AIC: [(4, 4, 260488.27142912982), (5, 4, 263264.96055363037), (4, 5, 260274.90755926829), (5, 5, 261622.0288602885), (4, 6, 260688.39772746345), (5, 6, 260912.61739148491)].
Best K,L for fold 2: (4, 5).
Performance: {'R^2': 0.7849633576236031, 'MSE': 2.5026825786772009, 'Rp': 0.88614872134170442}.

All model fits for fold 3, metric AIC: [(4, 4, 264054.08623711637), (5, 4, 262806.79434115923), (4, 5, 260829.63729924717), (5, 5, 260582.07428581853), (6, 5, 260368.48302208679), (5, 6, 261003.79373016706), (6, 6, 258770.32484170242), (7, 6, 256477.54559461726), (6, 7, 257521.93464259332), (7, 7, 255691.8631798806), (8, 7, 255839.06730957987), (7, 8, 254914.70496735093), (8, 8, 252414.47280171167)].
Best K,L for fold 3: (8, 8).
Performance: {'R^2': 0.799400251360757, 'MSE': 2.3896495579632071, 'Rp': 0.89434952909414756}.

All model fits for fold 4, metric AIC: [(4, 4, 259837.14859489043), (5, 4, 264873.83666251029), (4, 5, 260496.69826487306), (5, 5, 262240.62301195401)].
Best K,L for fold 4: (4, 4).
Performance: {'R^2': 0.7703783035434998, 'MSE': 2.6720517471725858, 'Rp': 0.87783310792893721}.

All model fits for fold 5, metric AIC: [(4, 4, 263992.13652410917), (5, 4, 263273.31627463415), (4, 5, 263337.09231507429), (5, 5, 263029.48140331329), (6, 5, 260083.73672747854), (5, 6, 259203.69263541372), (6, 6, 258740.37602282356), (7, 6, 258827.64802124051), (6, 7, 258515.61071871387), (7, 7, 256067.27897295184), (8, 7, 254889.66795575302), (7, 8, 254565.94321860731), (8, 8, 254078.05048890522)].
Best K,L for fold 5: (8, 8).
Performance: {'R^2': 0.803044768704817, 'MSE': 2.2976775826671298, 'Rp': 0.89638076167212632}.

all_MSE = [2.500413159374383,2.5026825786772009,2.3896495579632071,2.6720517471725858,2.2976775826671298]
all_R2 = [0.7848296007416965,0.7849633576236031,0.799400251360757,0.7703783035434998,0.803044768704817]

Average MSE: 2.472 +- 0.126
Average R^2: 0.789 +- 0.012
"""