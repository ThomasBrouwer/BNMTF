"""
Test the performance of Gibbs sampling for recovering a toy dataset, where we
vary the fraction of entries that are missing.

We use the correct number of latent factors and same priors as used to generate the data.

I, J, K = 100, 50, 10
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmf_gibbs import bnmf_gibbs
from ml_helpers.code.mask import generate_M, calc_inverse_M

import numpy, matplotlib.pyplot as plt

##########

fractions_unknown = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]

input_folder = project_location+"BNMTF/example/generate_toy/bnmf/"

iterations = 2000
init = 'random'
I,J,K = 100, 50, 10 #20,10,3 #

alpha, beta = 1., 1.
lambdaU = numpy.ones((I,K))
lambdaV = numpy.ones((J,K))/2.    
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

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
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.initialise(init)
    BNMF.run(iterations)
    
    taus = BNMF.all_tau
    all_taus.append(taus)
    
    # Approximate the expectations
    (exp_U, exp_V, exp_tau) = BNMF.approx_expectation(burn_in,thinning)
    
    # Measure the performances
    performances = BNMF.predict(M_test,burn_in,thinning)
    all_performances.append(performances)
    
print "All performances versus fraction of entries missing: %s." % zip(fractions_unknown,all_performances)

'''
All performances versus fraction of entries missing: 
    [(0.1, {'R^2': 0.9921454405020191, 'MSE': 1.5400714505804483, 'Rp': 0.9961135617253799}), 
     (0.2, {'R^2': 0.9911386335643347, 'MSE': 1.6136231501335585, 'Rp': 0.99556067260698711}), 
     (0.3, {'R^2': 0.9895838528353407, 'MSE': 1.6932177761546994, 'Rp': 0.99478499889866945}), 
     (0.4, {'R^2': 0.9878531525010826, 'MSE': 2.1472553993061561, 'Rp': 0.99392327919808154}), 
     (0.5, {'R^2': 0.9842400585194657, 'MSE': 2.6212716028727727, 'Rp': 0.9921031732459934}), 
     (0.6, {'R^2': 0.9770611102580021, 'MSE': 3.8908012819922386, 'Rp': 0.98848001493734927}), 
     (0.7, {'R^2': 0.9384162598941842, 'MSE': 10.30881561182588, 'Rp': 0.96946300290582577}), 
     (0.8, {'R^2': 0.7018702769103564, 'MSE': 50.590578325606941, 'Rp': 0.84127450698768969}), 
     (0.9, {'R^2': 0.6173081425136743, 'MSE': 65.01506935310023, 'Rp': 0.78611127067071707})]
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
#[0.14373635782712518, 0.17190904074733043, 0.19943426989940219, 0.21467319819848943, 0.33897862021632874, 0.6229985820084063, 4.4345153093047891, 58.824620998308362]
#[0.9990278157101131, 0.9987109092532906, 0.9986621065260793, 0.9984237679884446, 0.9976315034428627, 0.995569455182919, 0.9686873007784057, 0.5984791846575044]


# And plot tau for each fraction, so we see whether that has converged
f2, axarr2 = plt.subplots(len(fractions_unknown), sharex=True)
x2 = range(1,len(all_taus[0])+1)
axarr2[0].set_title('Convergence of values')
for i,taus in enumerate(all_taus):
    axarr2[i].plot(x2, taus)
    axarr2[i].set_ylabel("Fraction %s" % fractions_unknown[i])
axarr2[len(fractions_unknown)-1].set_xlabel("Iterations")