"""
Run the nested cross-validation for the NMTF class, on the Sanger dataset.

Since we want to find co-clusters of significantly higher/lower drug sensitivity
values, we should use the unstandardised Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.nmf_np import NMF
from BNMTF.cross_validation.nested_matrix_cross_validation import MatrixNestedCrossValidation
from BNMTF.drug_sensitivity.load_data import load_Sanger


# Settings
standardised = False
train_config = {
    'iterations' : 3000,
    'init_UV' : 'exponential',
    'expo_prior' : 0.1
}
K_range = [6,7,8,9,10]
P = 4
no_folds = 5
output_file = "./results.txt"
files_nested_performances = ["./fold_%s.txt" % fold for fold in range(1,no_folds+1)]

# Construct the parameter search
parameter_search = [{'K':K} for K in K_range]

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised)

# Run the cross-validation framework
random.seed(42)
numpy.random.seed(9000)
nested_crossval = MatrixNestedCrossValidation(
    method=NMF,
    X=X_min,
    M=M,
    K=no_folds,
    P=5,
    parameter_search=parameter_search,
    train_config=train_config,
    file_performance=output_file,
    files_nested_performances=files_nested_performances
)
nested_crossval.run()

"""
Average performances: {'R^2': 0.7945475836591555, 'MSE': 2.402773897685726, 'Rp': 0.89201588072427906}. 
All performances: {'R^2': [0.7971434454089756, 0.7932911883973699, 0.7937366204674767, 0.7912048835312413, 0.7973617804907137], 'MSE': [2.3573186661044465, 2.405759948350799, 2.4571177036234162, 2.4296979090873569, 2.3639752612626115], 'Rp': [0.89358850102890097, 0.89151992775702293, 0.89176931419249772, 0.88990343193851773, 0.89329822870445619]}. 

Average MSE: 2.403 +- 0.038
Average R^2: 0.795 +- 0.002
"""