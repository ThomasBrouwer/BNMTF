"""
Test the performance of Variational Bayes for recovering a toy dataset, where 
we vary the fraction of entries that are missing, and the amount of noise.

We use the correct number of latent factors and same priors as used to generate the data.

I, J, K = 100, 50, 10
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmf_vb import bnmf_vb
from BNMTF.example.generate_toy.bnmf.generate_bnmf import generate_dataset
from ml_helpers.code.mask import generate_M, calc_inverse_M

import numpy, matplotlib.pyplot as plt

##########

fractions_unknown = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ]
noise_levels = [ 100., 10., 1., 0.1, 0.01 ] # values for tau

input_folder = project_location+"BNMTF/example/generate_toy/bnmf/"

iterations = 100#1000
I,J,K = 100, 50, 10

alpha, beta = 1., 1.
lambdaU = numpy.ones((I,K))
lambdaV = numpy.ones((J,K))/2.    
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

# For each tau, generate a dataset, and mask matrices
all_R = []
all_Ms = []
all_Ms_test = []
for noise in noise_levels:
    (_,_,_,_,R) = generate_dataset(I,J,K,lambdaU,lambdaV,alpha,beta)
    Ms = [ generate_M(I,J,fraction) for fraction in fractions_unknown ]
    Ms_test = [ calc_inverse_M(M) for M in Ms ]
    
    all_R.append(R)
    all_Ms.append(Ms)
    all_Ms_test.append(Ms_test)
    
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
        
print "Generated datasets and mask matrices."
        
# For each level of noise, and for each fraction of missing data, run the VB algorithm.
# Then plot this, and convergence of tau as well.
all_performances_per_noise = []
for noise,R in zip(noise_levels,all_R):
    all_performances = []    
    all_exp_taus = []
    
    for fraction,M,M_test in zip(fractions_unknown,Ms,Ms_test):
        print "Trying fraction %s with noise level %s." % (fraction,noise)
        
        # Run the Gibbs sampler
        BNMF = bnmf_vb(R,M,K,priors)
        BNMF.initialise()
        BNMF.run(iterations)
        
        all_exp_taus.append(BNMF.all_exp_tau)
        
        # Measure the performances
        performances = BNMF.predict(M_test)
        all_performances.append(performances)
        
    all_performances_per_noise.append(all_performances)
    
    # Plot the tau values to check convergence
    f2, axarr2 = plt.subplots(len(fractions_unknown), sharex=True)
    x2 = range(1,len(all_exp_taus[0])+1)
    axarr2[0].set_title('Convergence of expectations tau with noise level %s' % noise)
    for i,taus in enumerate(all_exp_taus):
        axarr2[i].plot(x2, taus)
        axarr2[i].set_ylabel("Fraction %s" % fractions_unknown[i])
    axarr2[len(fractions_unknown)-1].set_xlabel("Iterations")
    

print "All performances versus noise level and fraction of entries missing: %s." \
    % zip(noise_levels,[zip(fractions_unknown,all_performances) for all_performances in all_performances_per_noise])


# Plot the MSE and R^2 for each noise level, as the fraction of missing values varies.
# So we get n lines in each plot for n noise levels.
f, axarr = plt.subplots(3, sharex=True)
x = fractions_unknown
axarr[0].set_title('Performance versus fraction missing')
for noise,all_performances in zip(noise_levels,all_performances_per_noise):
    axarr[0].plot(x, [perf['MSE'] for perf in all_performances],label="Noise %s" % noise)
    axarr[0].legend(loc="upper left")
axarr[0].set_ylabel("MSE")
for noise,all_performances in zip(noise_levels,all_performances_per_noise):
    axarr[1].plot(x, [perf['R^2'] for perf in all_performances],label="Noise %s" % noise)    
axarr[1].set_ylabel("R^2")
for noise,all_performances in zip(noise_levels,all_performances_per_noise):
    axarr[2].plot(x, [perf['Rp'] for perf in all_performances],label="Noise %s" % noise) 
axarr[2].set_ylabel("Rp")
axarr[2].set_xlabel("Fraction missing")

print [perf['MSE'] for perf in all_performances]
print [perf['R^2'] for perf in all_performances]
