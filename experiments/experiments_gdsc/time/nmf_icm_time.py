"""
Run NMF ICM on the Sanger dataset.

We can plot the MSE, R2 and Rp as it converges, against time, on the entire dataset.

We give flat priors (1/10).
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.models.nmf_icm import nmf_icm
from BNMTF.data_drug_sensitivity.gdsc.load_data import load_gdsc

import numpy, random, scipy, matplotlib.pyplot as plt

##########

standardised = False #standardised Sanger or unstandardised

repeats = 10

iterations = 1000
init_UV = 'random'
I, J, K = 622,138,25

minimum_TN = 0.1

alpha, beta = 1., 1. #1., 1.
lambdaU = numpy.ones((I,K))/10.
lambdaV = numpy.ones((J,K))/10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

# Load in data
(_,R,M,_,_,_,_) = load_gdsc(standardised=standardised)


# Run the VB algorithm, <repeats> times
times_repeats = []
performances_repeats = []
for i in range(0,repeats):
    # Set all the seeds
    numpy.random.seed(0)
    
    # Run the classifier
    nmf = nmf_icm(R,M,K,priors) 
    nmf.initialise(init_UV)
    nmf.run(iterations,minimum_TN=minimum_TN)

    # Extract the performances and timestamps across all iterations
    times_repeats.append(nmf.all_times)
    performances_repeats.append(nmf.all_performances)

# Check whether seed worked: all performances should be the same
assert all(numpy.array_equal(performances, performances_repeats[0]) for performances in performances_repeats), \
    "Seed went wrong - performances not the same across repeats!"

# Print out the performances, and the average times
icm_all_times_average = list(numpy.average(times_repeats, axis=0))
icm_all_performances = performances_repeats[0]
print "icm_all_times_average = %s" % icm_all_times_average
print "icm_all_performances = %s" % icm_all_performances


# Print all time plots, the average, and performance vs iterations
plt.figure()
plt.title("Performance against time")
plt.ylim(0,10)
for times in times_repeats:
    plt.plot(times, icm_all_performances['MSE'])

plt.figure()
plt.title("Performance against average time")
plt.plot(icm_all_times_average, icm_all_performances['MSE'])
plt.ylim(0,10)

plt.figure()
plt.title("Performance against iteration")
plt.plot(icm_all_performances['MSE'])
plt.ylim(0,10)