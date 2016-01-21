"""
Run the nested cross-validation for the NMTF class, on the Sanger dataset.

Since we want to find co-clusters of significantly higher/lower drug sensitivity
values, we should use the unstandardised Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.nmf_np import NMF
from BNMTF.cross_validation.matrix_cross_validation import MatrixCrossValidation
from BNMTF.drug_sensitivity.load_data import load_Sanger


# Settings
standardised = False
train_config = {
    'iterations' : 1000,
    'init_UV' : 'exponential',
    'expo_prior' : 0.1
}
K_range = range(2,10+1,2)
no_folds = 5
output_file = "./results.txt"
files_nested_performances = ["./fold_%s.txt" % fold for fold in range(1,no_folds+1)]

# Construct the parameter search
parameter_search = [{'K':K} for K in K_range]

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised,sep=',')

# Run the cross-validation framework
random.seed(42)
numpy.random.seed(9000)
nested_crossval = MatrixCrossValidation(
    method=NMF,
    X=X_min,
    M=M,
    K=no_folds,
    parameter_search=parameter_search,
    train_config=train_config,
    file_performance=output_file
)
nested_crossval.run()

"""
Average performances: {'R^2': 0.7945475836591555, 'MSE': 2.402773897685726, 'Rp': 0.89201588072427906}. 
All performances: {'R^2': [0.7971434454089756, 0.7932911883973699, 0.7937366204674767, 0.7912048835312413, 0.7973617804907137], 'MSE': [2.3573186661044465, 2.405759948350799, 2.4571177036234162, 2.4296979090873569, 2.3639752612626115], 'Rp': [0.89358850102890097, 0.89151992775702293, 0.89176931419249772, 0.88990343193851773, 0.89329822870445619]}. 

nested_crossval.average_performances
{'{"K": 2}': {'R^2': 0.7574722793141616, 'MSE': 2.8478524479649194, 'Rp': 0.87038978703323477}, '{"K": 10}': {'R^2': 0.7968644308813252, 'MSE': 2.3851740431836719, 'Rp': 0.89360721885736627}, '{"K": 4}': {'R^2': 0.782291563558489, 'MSE': 2.5567566984337189, 'Rp': 0.88466324440659361}, '{"K": 6}': {'R^2': 0.7955822884834453, 'MSE': 2.4005934744766102, 'Rp': 0.8923009207438467}, '{"K": 8}': {'R^2': 0.799457981626682, 'MSE': 2.3547329318771193, 'Rp': 0.89473316906463174}}
"""