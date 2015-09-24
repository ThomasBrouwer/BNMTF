"""
Test the performance of Variational Bayes for recovering a toy dataset, where 
we vary the fraction of entries that are missing.

We use the correct number of latent factors and same priors as used to generate the data.

I, J, K = 100, 50, 10

When 80% of the values are missing, we converge to a local minimum, but one
that has very poor performance (MSE 13 on the training, 45 on test - but still 
an R^2 of 0.85). 
So we want to know at what fraction the performance drops.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmf_vb import bnmf_vb
from ml_helpers.code.mask import generate_M, calc_inverse_M

import numpy, matplotlib.pyplot as plt

##########

fractions_unknown = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ]

input_folder = project_location+"BNMTF/example/generate_toy/bnmf/"

iterations = 1000
I,J,K = 100, 50, 10 #20,10,3 #

alpha, beta = 10., 1.
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


# For each M, run the VB algorithm.
# From informal experiments, the VB algorithm seems to converge around:
# (Missing -> Convergence): 0.1 -> 400, 0.5 -> TODO: , 0.8 -> TODO:
all_performances = []
all_exp_taus = []
for i,(M,M_test) in enumerate(zip(Ms,Ms_test)):
    print "Trying fraction %s." % fractions_unknown[i]
    
    # Run the Gibbs sampler
    BNMF = bnmf_vb(R,M,K,priors)
    BNMF.initialise()
    BNMF.run(iterations)
    
    all_exp_taus.append(BNMF.all_exp_tau)
    
    # Measure the performances
    performances = BNMF.predict(M_test)
    all_performances.append(performances)
    
print "All performances versus fraction of entries missing: %s." % zip(fractions_unknown,all_performances)

# All performances versus fraction of entries missing: 
#   [(0.1, {'R^2': 0.9982983596866993, 'MSE': 0.15079790280194472, 'Rp': 0.99915702753129576}), 
#    (0.2, {'R^2': 0.9976870639220524, 'MSE': 0.16479880795721666, 'Rp': 0.99884905432714388}), 
#    (0.3, {'R^2': 0.9978980032076648, 'MSE': 0.17617546650177018, 'Rp': 0.99894894652953148}), 
#    (0.4, {'R^2': 0.9975688668219221, 'MSE': 0.21225423755771641, 'Rp': 0.99878386898437976}), 
#    (0.5, {'R^2': 0.99687220093907, 'MSE': 0.26155176773507094, 'Rp': 0.99843650708612997}), 
#    (0.6, {'R^2': 0.9937541894447948, 'MSE': 0.52022196910172147, 'Rp': 0.99689034359616391}), 
#    (0.7, {'R^2': 0.8374553590021275, 'MSE': 13.634768188449405, 'Rp': 0.91704136046476603}), 
#    (0.8, {'R^2': 0.4579653310318519, 'MSE': 44.705591667008086, 'Rp': 0.69504430980172005})]

# Plot the MSE and R^2
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


# And plot tau for each fraction, so we see whether that has converged
f2, axarr2 = plt.subplots(len(fractions_unknown), sharex=True)
x2 = range(1,len(all_exp_taus[0])+1)
axarr2[0].set_title('Convergence of expectations tau')
for i,taus in enumerate(all_exp_taus):
    axarr2[i].plot(x2, taus)
    axarr2[i].set_ylabel("Fraction %s" % fractions_unknown[i])
axarr2[len(fractions_unknown)-1].set_xlabel("Iterations")