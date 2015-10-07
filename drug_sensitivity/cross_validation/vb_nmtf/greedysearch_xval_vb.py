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
All model fits for fold 1, metric AIC: [(4, 4, 260452.11985016626), (5, 4, 261929.48771940501), (4, 5, 261430.22590459499), (5, 5, 267036.0484551065)].
Best K,L for fold 1: (4, 4).
Performance: {'R^2': 0.787667228903906, 'MSE': 2.4674381645672101, 'Rp': 0.88751155893742173}.

All model fits for fold 2, metric AIC: [(4, 4, 265652.55714846723), (5, 4, 261419.11268396635), (4, 5, 260816.33930344926), (5, 5, 263234.80643280019), (4, 6, 266255.85560067318), (5, 6, 257763.80296080254), (6, 6, 258911.57711492985), (5, 7, 262095.28260532857), (6, 7, 259234.55022118741)].
Best K,L for fold 2: (5, 6).
Performance: {'R^2': 0.7935264093167969, 'MSE': 2.4030223530708295, 'Rp': 0.89096151692324177}.

All model fits for fold 3, metric AIC: [(4, 4, 265832.40347713884), (5, 4, 261153.32969799257), (4, 5, 260600.07447752019), (5, 5, 261887.53424325009), (4, 6, 260892.97417301565), (5, 6, 257665.38964746773), (6, 6, 265013.25810553401), (5, 7, 257649.80957092362), (6, 7, 263559.21339785459), (5, 8, 262667.45644866489), (6, 8, 263978.58577695128)].
Best K,L for fold 3: (5, 7).
Performance: {'R^2': 0.7645025959725491, 'MSE': 2.8053687567063563, 'Rp': 0.87442844285002508}.

All model fits for fold 4, metric AIC: [(4, 4, 260111.40809577561), (5, 4, 261428.71715128978), (4, 5, 260704.67969425328), (5, 5, 261619.94914788904)].
Best K,L for fold 4: (4, 4).
Performance: {'R^2': 0.7726773856386991, 'MSE': 2.6452978888735972, 'Rp': 0.87907804976475812}.

All model fits for fold 5, metric AIC: [(4, 4, 260663.72097505987), (5, 4, 261465.16869146327), (4, 5, 261258.26836494287), (5, 5, 257420.75689308817), (6, 5, 259272.00566491604), (5, 6, 258211.19280392904), (6, 6, 268541.1457820338)].
Best K,L for fold 5: (5, 5).
Performance: {'R^2': 0.776552440362962, 'MSE': 2.606736796496838, 'Rp': 0.88125592443305967}.

all_MSE = [2.4674381645672101,2.4030223530708295,2.8053687567063563,2.6452978888735972,2.606736796496838]
all_R2 = [0.787667228903906,0.7935264093167969,0.7645025959725491,0.7726773856386991,0.776552440362962]

Average MSE: 2.586 +- 0.141
Average R^2: 0.779 +- 0.010
"""