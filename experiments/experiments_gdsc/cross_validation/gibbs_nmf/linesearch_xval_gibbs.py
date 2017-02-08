"""
Run the cross validation with line search for model selection using G-NMF on
the Sanger dataset.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

import numpy, random
from BNMTF.code.models.bnmf_gibbs_optimised import bnmf_gibbs_optimised
from BNMTF.code.cross_validation.line_search_cross_validation import LineSearchCrossValidation
from BNMTF.experiments.experiments_gdsc.load_data import load_gdsc


# Settings
standardised = False
iterations = 1000
burn_in = 900
thinning = 2
init_UV = 'random'

K_range = [15,20,25,30]
no_folds = 10
restarts = 1

quality_metric = 'AIC'
output_file = "./results.txt"

alpha, beta = 1., 1.
lambdaU = 1./10.
lambdaV = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_gdsc(standardised=standardised)

# Run the cross-validation framework
#random.seed(42)
#numpy.random.seed(9000)
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
all_MSE = [2.0115451703143985, 2.0532542729784833, 2.0454971069846226, 1.994656076757727, 2.0281421630490297, 2.0691704067461281, 2.0708801136454622, 2.1137440615703653, 2.1153688464049725, 2.0478097531374373]
all_R2 = [0.8248485588294542, 0.8219514639515233, 0.8217549958515522, 0.8349672123366683, 0.830543344804296, 0.8229475100079148, 0.8234388009582426, 0.8228191950789238, 0.8195240616800068, 0.8266748390223762, ]

Average MSE: 2.0550067971588626 +- 0.0013944347250178673
Average R^2: 0.82494699825209561 +- 1.9408941387580883e-05
"""