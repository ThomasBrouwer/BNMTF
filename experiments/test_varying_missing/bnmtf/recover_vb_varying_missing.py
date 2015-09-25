"""
Test the performance of variational Bayes for recovering a toy dataset, where 
we vary the fraction of entries that are missing.

We use the correct number of latent factors and same priors as used to generate the data.

I, J, K, L = 100, 50, 10, 5
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmtf_vb_optimised import bnmtf_vb_optimised
from BNMTF.experiments.generate_toy.bnmtf.generate_bnmtf import try_generate_M
from ml_helpers.code.mask import calc_inverse_M

import numpy, matplotlib.pyplot as plt

##########

fractions_unknown = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]

input_folder = project_location+"BNMTF/experiments/generate_toy/bnmtf/"

repeats = 10 # number of times we try each fraction
iterations = 1000
init = 'random'
I,J,K,L = 100, 80, 5, 5

alpha, beta = 1., 1.
lambdaF = numpy.ones((I,K))/10.
lambdaS = numpy.ones((K,L))/10.
lambdaG = numpy.ones((J,L))/10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

init_S = 'random'
init_FG = 'kmeans'

metrics = ['MSE', 'R^2', 'Rp']

# Load in data
R = numpy.loadtxt(input_folder+"R.txt")

# Generate matrices M - one list of M's for each fraction
M_attempts = 100
all_Ms = [ 
    [try_generate_M(I,J,fraction,M_attempts) for r in range(0,repeats)]
    for fraction in fractions_unknown
]
all_Ms_test = [ [calc_inverse_M(M) for M in Ms] for Ms in all_Ms ]

# Make sure each M has no empty rows or columns
def check_empty_rows_columns(M,fraction):
    sums_columns = M.sum(axis=0)
    sums_rows = M.sum(axis=1)
    for i,c in enumerate(sums_rows):
        assert c != 0, "Fully unobserved row in M, row %s. Fraction %s." % (i,fraction)
    for j,c in enumerate(sums_columns):
        assert c != 0, "Fully unobserved column in M, column %s. Fraction %s." % (j,fraction)
        
for Ms,fraction in zip(all_Ms,fractions_unknown):
    for M in Ms:
        check_empty_rows_columns(M,fraction)


# We now run the VB algorithm on each of the M's for each fraction.
all_performances = {metric:[] for metric in metrics} 
average_performances = {metric:[] for metric in metrics} # averaged over repeats
for (fraction,Ms,Ms_test) in zip(fractions_unknown,all_Ms,all_Ms_test):
    print "Trying fraction %s." % fraction
    
    # Run the algorithm <repeats> times and store all the performances
    for metric in metrics:
        all_performances[metric].append([])
    for (repeat,M,M_test) in zip(range(0,repeats),Ms,Ms_test):
        print "Repeat %s of fraction %s." % (repeat+1, fraction)
    
        # Run the VB algorithm
        BNMTF = bnmtf_vb_optimised(R,M,K,L,priors)
        BNMTF.initialise(init_S,init_FG)
        BNMTF.run(iterations)
    
        # Measure the performances
        performances = BNMTF.predict(M_test)
        for metric in metrics:
            # Add this metric's performance to the list of <repeat> performances for this fraction
            all_performances[metric][-1].append(performances[metric])
    
    # Compute the average across attempts
    for metric in metrics:
        average_performances[metric].append(sum(all_performances[metric][-1])/repeats)
        

print "repeats=%s \nfractions_unknown = %s \nall_performances = %s \naverage_performances = %s" % \
    (repeats,fractions_unknown,all_performances,average_performances)


'''

'''


# Plot the MSE, R^2 and Rp
for metric in metrics:
    plt.figure()
    x = fractions_unknown
    y = average_performances[metric]
    plt.plot(x,y)
    plt.xlabel("Fraction missing")
    plt.ylabel(metric)