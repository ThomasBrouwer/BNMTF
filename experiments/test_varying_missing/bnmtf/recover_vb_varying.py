"""
Test the performance of variational Bayes for recovering a toy dataset, where 
we vary the fraction of entries that are missing.

We use the correct number of latent factors and same priors as used to generate the data.

I, J, K, L = 100, 50, 10, 5
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmtf_vb import bnmtf_vb
from BNMTF.example.generate_toy.bnmtf.generate_bnmtf import try_generate_M
from ml_helpers.code.mask import calc_inverse_M

import numpy, matplotlib.pyplot as plt

##########

fractions_unknown = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]

input_folder = project_location+"BNMTF/example/generate_toy/bnmtf/"

iterations = 1000
init = 'random'
I,J,K,L = 50, 40, 10, 5

alpha, beta = 1., 1.
lambdaF = numpy.ones((I,K))
lambdaS = numpy.ones((K,L))   
lambdaG = numpy.ones((J,L))   
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

# Load in data
R = numpy.loadtxt(input_folder+"R.txt")

# Generate matrices M
Ms = [ try_generate_M(I,J,fraction,attempts=100) for fraction in fractions_unknown ]
Ms_test = [ calc_inverse_M(M) for M in Ms ]

# Make sure each M has no empty rows or columns
def check_empty_rows_columns(M,fraction):
    sums_columns = M.sum(axis=0)
    sums_rows = M.sum(axis=1)
    for i,c in enumerate(sums_rows):
        assert c != 0, "Fully unobserved row in M, row %s. Fraction %s." % (i,fraction)
    for j,c in enumerate(sums_columns):
        assert c != 0, "Fully unobserved column in M, column %s. Fraction %s." % (j,fraction)
        
for M,fraction in zip(Ms,fractions_unknown):
    check_empty_rows_columns(M,fraction)


# For each M, run the VB algorithm.
all_performances = []
all_taus = []
for i,(M,M_test) in enumerate(zip(Ms,Ms_test)):
    print "Trying fraction %s." % fractions_unknown[i]
    
    # Run the Gibbs sampler
    BNMTF = bnmtf_vb(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.run(iterations)
    
    taus = BNMTF.all_exp_tau
    all_taus.append(taus)
    
    # Measure the performances
    performances = BNMTF.predict(M_test)
    all_performances.append(performances)
    
print "All performances versus fraction of entries missing: %s." % zip(fractions_unknown,all_performances)


'''
All performances versus fraction of entries missing: 
    
'''


# Also plot the MSE and R^2
f, axarr = plt.subplots(3, sharex=True)
x = fractions_unknown
axarr[0].set_title('Performance versus fraction missing')
axarr[0].plot(x, [perf['MSE'] for perf in all_performances])
axarr[0].set_ylabel("MSE")
axarr[1].plot(x, [perf['R^2'] for perf in all_performances])    
axarr[1].set_ylabel("R^2")
axarr[2].plot(x, [perf['Rp'] for perf in all_performances]) 
axarr[2].set_ylabel("Rp")
axarr[2].set_xlabel("Fraction missing")

print [perf['MSE'] for perf in all_performances]
print [perf['R^2'] for perf in all_performances]
print [perf['Rp'] for perf in all_performances]


# And plot tau for each fraction, so we see whether that has converged
f2, axarr2 = plt.subplots(len(fractions_unknown), sharex=True)
x2 = range(1,len(all_taus[0])+1)
axarr2[0].set_title('Convergence of values (tau)')
for i,taus in enumerate(all_taus):
    axarr2[i].plot(x2, taus)
    axarr2[i].set_ylabel("Fraction %s" % fractions_unknown[i])
axarr2[len(fractions_unknown)-1].set_xlabel("Iterations")