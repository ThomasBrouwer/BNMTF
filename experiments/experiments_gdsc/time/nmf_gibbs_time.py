"""
Run NMF Gibbs on the Sanger dataset.

We can plot the MSE, R2 and Rp as it converges, against time, on the entire dataset.

We give flat priors (1/10).
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BNMTF.code.models.bnmf_gibbs_optimised import bnmf_gibbs_optimised
from BNMTF.data_drug_sensitivity.gdsc.load_data import load_gdsc

import numpy, random, scipy, matplotlib.pyplot as plt

##########

standardised = False #standardised Sanger or unstandardised

repeats = 10

iterations = 500
init_UV = 'random'
I, J, K = 622,138,25

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
    BNMF = bnmf_gibbs_optimised(R,M,K,priors) 
    BNMF.initialise(init_UV)
    BNMF.run(iterations)

    # Extract the performances and timestamps across all iterations
    times_repeats.append(BNMF.all_times)
    performances_repeats.append(BNMF.all_performances)

# Check whether seed worked: all performances should be the same
assert all(numpy.array_equal(performances, performances_repeats[0]) for performances in performances_repeats), \
    "Seed went wrong - performances not the same across repeats!"

# Print out the performances, and the average times
gibbs_all_times_average = list(numpy.average(times_repeats, axis=0))
gibbs_all_performances = performances_repeats[0]
print "gibbs_all_times_average = %s" % gibbs_all_times_average
print "gibbs_all_performances = %s" % gibbs_all_performances

# Print all time plots, the average, and performance vs iterations
plt.figure()
plt.title("Performance against time")
plt.ylim(0,10)
for times in times_repeats:
    plt.plot(times, gibbs_all_performances['MSE'])

plt.figure()
plt.title("Performance against average time")
plt.plot(gibbs_all_times_average, gibbs_all_performances['MSE'])
plt.ylim(0,10)

plt.figure()
plt.title("Performance against iteration")
plt.plot(gibbs_all_performances['MSE'])
plt.ylim(0,10)