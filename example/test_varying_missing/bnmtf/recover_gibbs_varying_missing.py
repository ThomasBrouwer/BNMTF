"""
Test the performance of Gibbs sampling for recovering a toy dataset, where we
vary the fraction of entries that are missing.

We use the correct number of latent factors and same priors as used to generate the data.

I, J, K, L = 100, 50, 10, 5
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmtf_gibbs import bnmtf_gibbs
from ml_helpers.code.mask import generate_M, calc_inverse_M

import numpy, matplotlib.pyplot as plt

##########

fractions_unknown = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]

input_folder = project_location+"BNMTF/example/generate_toy/bnmtf/"

iterations = 2000
init = 'random'
I,J,K,L = 100, 50, 10, 5

alpha, beta = 1., 1.
lambdaF = numpy.ones((I,K))
lambdaS = numpy.ones((K,L))*2.   
lambdaG = numpy.ones((J,L))   
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

# Load in data
R = numpy.loadtxt(input_folder+"R.txt")

# Generate matrices M
Ms = [ generate_M(I,J,fraction) for fraction in fractions_unknown ]
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


# For each M, run the Gibbs sampler.
# From informal experiments, the Gibbs sampler seems to converge around:
# (Missing -> Burn_in): 0.1 -> 400, 0.5 -> 400 , 0.8 -> 1000
burn_in = 1000
thinning = 10

all_performances = []
all_taus = []
for i,(M,M_test) in enumerate(zip(Ms,Ms_test)):
    print "Trying fraction %s." % fractions_unknown[i]
    
    # Run the Gibbs sampler
    BNMTF = bnmtf_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.run(iterations)
    
    taus = BNMTF.all_tau
    all_taus.append(taus)
    
    # Approximate the expectations
    (exp_F, exp_S, exp_G, exp_tau) = BNMTF.approx_expectation(burn_in,thinning)
    
    # Measure the performances
    performances = BNMTF.predict(M_test,burn_in,thinning)
    all_performances.append(performances)
    
print "All performances versus fraction of entries missing: %s." % zip(fractions_unknown,all_performances)


'''
All performances versus fraction of entries missing: 
    [(0.1, {'R^2': 0.9917750383486127, 'MSE': 1.2277250194175178, 'Rp': 0.99591544694678946}), 
     (0.2, {'R^2': 0.9895463791069684, 'MSE': 1.615141222475412, 'Rp': 0.99489139990761932}), 
     (0.3, {'R^2': 0.9900547853371741, 'MSE': 1.5293654805813586, 'Rp': 0.99510663448111747}), 
     (0.4, {'R^2': 0.9890688014092863, 'MSE': 1.6048103637988647, 'Rp': 0.99476122991038363}), 
     (0.5, {'R^2': 0.9887851334327311, 'MSE': 1.613840918172319, 'Rp': 0.99443493581791065}), 
     (0.6, {'R^2': 0.9876175724086197, 'MSE': 1.795215154336347, 'Rp': 0.99389763265210129}), 
     (0.7, {'R^2': 0.9862225086827572, 'MSE': 2.0617018055912903, 'Rp': 0.99312105712276122}), 
     (0.8, {'R^2': 0.9782326573540902, 'MSE': 3.2358943173672592, 'Rp': 0.98920368952036386}), 
     (0.9, {'R^2': 0.9550313758299558, 'MSE': 6.6583583251909966, 'Rp': 0.97754790904654032})].
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
axarr2[0].set_title('Convergence of values')
for i,taus in enumerate(all_taus):
    axarr2[i].plot(x2, taus)
    axarr2[i].set_ylabel("Fraction %s" % fractions_unknown[i])
axarr2[len(fractions_unknown)-1].set_xlabel("Iterations")