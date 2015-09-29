"""
Run the line search method for finding the best value for K for BNMF.
We use the parameters for the true priors.

The BIC tends to give overly simple models, preferring K=1 oftentimes.
The log likelihood and AIC tend to peak at the true K if the correct priors are
given (this has to do with converging to a good local minimum).

If we give the wrong prior (true/5) we still obtain good convergence (with 
true*5 all values get pushed to 0, leading to terrible solutions), and we get
a nice peak for the log likelihood and AIC around the true K.
"""


project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.experiments.generate_toy.bnmf.generate_bnmf import generate_dataset, try_generate_M
from BNMTF.grid_search.line_search_bnmf import LineSearch
from BNMTF.code.bnmf_vb_optimised import bnmf_vb_optimised

import numpy, matplotlib.pyplot as plt

##########

restarts = 5
iterations = 1000

I, J = 100, 80
true_K = 10
values_K = range(1,20+1)

fraction_unknown = 0.1
attempts_M = 100

alpha, beta = 1., 1. #1., 1.
tau = alpha / beta
lambdaU = numpy.ones((I,true_K))
lambdaV = numpy.ones((J,true_K))

classifier = bnmf_vb_optimised
initUV = 'random'

# Generate data
(_,_,_,_,R) = generate_dataset(I,J,true_K,lambdaU,lambdaV,tau)
M = numpy.ones((I,J))
#M = try_generate_M(I,J,fraction_unknown,attempts_M)

# Run the line search. The priors lambdaU and lambdaV need to be a single value (recall K is unknown)
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU[0,0]/10, 'lambdaV':lambdaV[0,0]/10 }
line_search = LineSearch(classifier,values_K,R,M,priors,initUV,iterations,restarts)
line_search.search()

# Plot the performances of all three metrics
metrics = ['loglikelihood', 'BIC', 'AIC', 'MSE']
for metric in metrics:
    plt.figure()
    plt.plot(values_K, line_search.all_values(metric), label=metric)
    plt.legend(loc=3)
    
# Also print out all values in a dictionary
all_values = {}
for metric in metrics:
    all_values[metric] = line_search.all_values(metric)
    
print "all_values = %s" % all_values

'''
all_values = {'MSE': [9.6549057058769172, 6.9884484321573099, 5.1934558411893263, 4.0992884584119622, 3.2240598340019346, 2.4818906465963311, 1.8406504508732986, 1.3502204910226565, 1.0333349580316775, 0.78017988120303972, 0.75361070313227285, 0.72978226838232485, 0.7099944773500656, 0.69208188151852712, 0.6725243354242475, 0.65582783054509486, 0.63850413370173831, 0.62299533285704889, 0.60796336865454281, 0.59211139754569109], 'loglikelihood': [-20422.878028449726, -19133.050411760163, -17950.489043043988, -17011.744324390602, -16060.301690117441, -15024.660282332354, -13842.706832984462, -12619.297660607001, -11566.289337162721, -10461.721305930792, -10340.456751510523, -10230.967539005962, -10140.60032716368, -10057.810701562652, -9965.5778556505502, -9887.236373074742, -9803.3442204645653, -9729.0856510921694, -9657.9307609909029, -9581.8361432246875], 'AIC': [41205.756056899452, 38986.100823520326, 36980.978086087976, 35463.488648781204, 33920.603380234883, 32209.320564664707, 30205.413665968925, 28118.595321214001, 26372.578674325443, 24523.442611861585, 24640.913503021045, 24781.935078011924, 24961.20065432736, 25155.621403125304, 25331.1557113011, 25534.472746149484, 25726.688440929131, 25938.171302184339, 26155.861521981806, 26363.672286449375], 'BIC': [42463.451484618607, 41501.491678958635, 40754.06436924544, 40494.270359657821, 40209.080518830655, 39755.493130979637, 39009.281660003013, 38180.158742967244, 37691.83752379784, 37100.39688905314, 38475.563207931751, 39874.28021064178, 41311.241214676382, 42763.35739119348, 44196.587127088424, 45657.599589655962, 47107.510712154763, 48576.689001129133, 50052.07464864575, 51517.580840832481]}
'''