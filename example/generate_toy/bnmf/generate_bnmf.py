"""
Generate a toy dataset for the matrix factorisation case, and store it.

We use dimensions 100 by 50 for the dataset, and 10 latent factors.

As the prior for U we take value 1 for all entries (so exp 1).
As the prior for V we take value 0.5 for all entries (so exp 2).

As a result, each value in R has a value of around 40.

We add Gaussian noise of precision tau = 1 (prior for gamma: alpha=1,beta=1).
(Simply using the expectation of our Gamma distribution over tau)
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.distributions.exponential import Exponential
from BNMTF.code.distributions.normal import Normal
from ml_helpers.code.mask import generate_M

import numpy, itertools

##########

output_folder = project_location+"BNMTF/example/generate_toy/bnmf/"

I,J,K = 100, 50, 10 #20,10,3 # 
fraction_unknown = 0.1
alpha, beta = 10., 1.
lambdaU = numpy.ones((I,K))
lambdaV = numpy.ones((I,K))/2.

# Generate U, V
U = numpy.zeros((I,K))
for i,k in itertools.product(xrange(0,I),xrange(0,K)):
    U[i,k] = Exponential(lambdaU[i,k]).draw()
V = numpy.zeros((I,K))
for j,k in itertools.product(xrange(0,J),xrange(0,K)):
    V[j,k] = Exponential(lambdaV[j,k]).draw()
tau = alpha / beta

# Generate R
true_R = numpy.dot(U,V.T)
R = numpy.zeros((I,J))
for i,j in itertools.product(xrange(0,I),xrange(0,J)):
    R[i,j] = Normal(true_R[i,j],tau).draw()
    
# Make a mask matrix M
M = generate_M(I,J,fraction_unknown)
    
# Store all matrices in text files
numpy.savetxt(open(output_folder+"U.txt",'w'),U)
numpy.savetxt(open(output_folder+"V.txt",'w'),V)
numpy.savetxt(open(output_folder+"R_true.txt",'w'),true_R)
numpy.savetxt(open(output_folder+"R.txt",'w'),R)
numpy.savetxt(open(output_folder+"M.txt",'w'),M)