"""
Generate a toy dataset for the matrix factorisation case, and store it.

We use dimensions 100 by 50 for the dataset, and 10 latent factors.

As the prior for U and V we take value 1 for all entries (so exp 1).

As a result, each value in R has a value of around 20, and a variance of 100-120.

For contrast, the Sanger dataset of 705 by 140 shifted to nonnegative has mean 
31.522999753779082 and variance 243.2427345740027.

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

def generate_dataset(I,J,K,lambdaU,lambdaV,tau):
    # Generate U, V
    U = numpy.zeros((I,K))
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        U[i,k] = Exponential(lambdaU[i,k]).draw()
    V = numpy.zeros((J,K))
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        V[j,k] = Exponential(lambdaV[j,k]).draw()
    
    # Generate R
    true_R = numpy.dot(U,V.T)
    
    R = numpy.zeros((I,J))
    for i,j in itertools.product(xrange(0,I),xrange(0,J)):
        R[i,j] = Normal(true_R[i,j],tau).draw()   
        
    return (U,V,tau,true_R,R)
    
def add_noise(true_R,tau):
    (I,J) = true_R.shape
    R = numpy.zeros((I,J))
    for i,j in itertools.product(xrange(0,I),xrange(0,J)):
        R[i,j] = Normal(true_R[i,j],tau).draw()
    return R
    
##########

if __name__ == "__main__":
    output_folder = project_location+"BNMTF/example/generate_toy/bnmf/"

    I,J,K = 100, 50, 10 #20,10,3 # 705, 140, 15 #
    fraction_unknown = 0.9
    alpha, beta = 1., 1.
    lambdaU = numpy.ones((I,K))
    lambdaV = numpy.ones((I,K))/2.
    tau = alpha / beta
    
    (U,V,tau,true_R,R) = generate_dataset(I,J,K,lambdaU,lambdaV,tau)
    M = generate_M(I,J,fraction_unknown)
    
    # Store all matrices in text files
    numpy.savetxt(open(output_folder+"U.txt",'w'),U)
    numpy.savetxt(open(output_folder+"V.txt",'w'),V)
    numpy.savetxt(open(output_folder+"R_true.txt",'w'),true_R)
    numpy.savetxt(open(output_folder+"R.txt",'w'),R)
    numpy.savetxt(open(output_folder+"M.txt",'w'),M)
    
    print "Mean R: %s. Variance R: %s. Min R: %s. Max R: %s." % (numpy.mean(R),numpy.var(R),R.min(),R.max())