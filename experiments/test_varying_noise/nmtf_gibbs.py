"""
Test the performance of Gibbs sampling for recovering a toy dataset, where we
vary the level of noise.
We repeat this 10 times per fraction and average that.

We use the correct number of latent factors and flatter priors than used to generate the data.

I, J, K, L = 100, 80, 5, 5

The noise levels indicate the percentage of noise, compared to the amount of 
variance in the dataset - i.e. the inverse of the Signal to Noise ratio:
    SNR = std_signal^2 / std_noise^2
    noise = 1 / SNR
We test it for 1%, 2%, 5%, 10%, 20%, 50% noise.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmtf_gibbs_optimised import bnmtf_gibbs_optimised
from BNMTF.experiments.generate_toy.bnmtf.generate_bnmtf import add_noise, try_generate_M
from ml_helpers.code.mask import calc_inverse_M

import numpy, matplotlib.pyplot as plt

##########

fraction_unknown = 0.1
noise_ratios = [ 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 ] # 1/SNR

input_folder = project_location+"BNMTF/experiments/generate_toy/bnmtf/"

repeats = 10
iterations = 2000
burn_in = 1800
thinning = 5

init_S = 'random'
init_FG = 'kmeans'
I,J,K,L = 100, 80, 5, 5

alpha, beta = 1., 1.
lambdaF = numpy.ones((I,K))/10.
lambdaS = numpy.ones((K,L))/10.
lambdaG = numpy.ones((J,L))/10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

metrics = ['MSE', 'R^2', 'Rp']


# Load in data
R_true = numpy.loadtxt(input_folder+"R_true.txt")


# For each noise ratio, generate mask matrices for each attempt
M_attempts = 100
all_Ms = [ 
    [try_generate_M(I,J,fraction_unknown,M_attempts) for r in range(0,repeats)]
    for noise in noise_ratios
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
        
for Ms in all_Ms:
    for M in Ms:
        check_empty_rows_columns(M,fraction_unknown)


# For each noise ratio, add that level of noise to the true R
all_R = []
variance_signal = R_true.var()
for noise in noise_ratios:
    tau = 1. / (variance_signal * noise)
    print "Noise: %s%%. Variance in dataset is %s. Adding noise with variance %s." % (100.*noise,variance_signal,1./tau)
    
    R = add_noise(R_true,tau)
    all_R.append(R)
    
    
# We now run the VB algorithm on each of the M's for each noise ratio    
all_performances = {metric:[] for metric in metrics} 
average_performances = {metric:[] for metric in metrics} # averaged over repeats
for (noise,R,Ms,Ms_test) in zip(noise_ratios,all_R,all_Ms,all_Ms_test):
    print "Trying noise ratio %s." % noise
    
    # Run the algorithm <repeats> times and store all the performances
    for metric in metrics:
        all_performances[metric].append([])
    for (repeat,M,M_test) in zip(range(0,repeats),Ms,Ms_test):
        print "Repeat %s of noise ratio %s." % (repeat+1, noise)
    
        BNMF = bnmtf_gibbs_optimised(R,M,K,L,priors)
        BNMF.initialise(init_S,init_FG)
        BNMF.run(iterations)
    
        # Measure the performances
        performances = BNMF.predict(M_test,burn_in,thinning)
        for metric in metrics:
            # Add this metric's performance to the list of <repeat> performances for this noise ratio
            all_performances[metric][-1].append(performances[metric])
            
    # Compute the average across attempts
    for metric in metrics:
        average_performances[metric].append(sum(all_performances[metric][-1])/repeats)
    

    
print "repeats=%s \nnoise_ratios = %s \nall_performances = %s \naverage_performances = %s" % \
    (repeats,noise_ratios,all_performances,average_performances)


'''

'''


# Plot the MSE, R^2 and Rp
for metric in metrics:
    plt.figure()
    x = noise_ratios
    y = average_performances[metric]
    plt.plot(x,y)
    plt.xlabel("Noise ratios missing")
    plt.ylabel(metric)