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

import numpy, itertools, matplotlib.pyplot as plt

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
    R = add_noise(true_R,tau)    
    
    return (U,V,tau,true_R,R)
    
def add_noise(true_R,tau):
    if numpy.isinf(tau):
        return numpy.copy(true_R)
    
    (I,J) = true_R.shape
    R = numpy.zeros((I,J))
    for i,j in itertools.product(xrange(0,I),xrange(0,J)):
        R[i,j] = Normal(true_R[i,j],tau).draw()
    return R
    
##########

if __name__ == "__main__":
    output_folder = project_location+"BNMTF/example/generate_toy/bnmf/"

    I,J,K = 20,10,5#100, 50, 10
    fraction_unknown = 0.8
    alpha, beta = 100., 1.
    lambdaU = numpy.ones((I,K))
    lambdaV = numpy.ones((I,K))
    tau = alpha / beta
    
    (U,V,tau,true_R,R) = generate_dataset(I,J,K,lambdaU,lambdaV,tau)
    
    # Try to generate M
    attempts = 10000
    for i in range(0,attempts):
        try:
            M = generate_M(I,J,fraction_unknown)
            sums_columns = M.sum(axis=0)
            sums_rows = M.sum(axis=1)
            for i,c in enumerate(sums_rows):
                assert c != 0, "Fully unobserved row in M, row %s. Fraction %s." % (i,fraction_unknown)
            for j,c in enumerate(sums_columns):
                assert c != 0, "Fully unobserved column in M, column %s. Fraction %s." % (j,fraction_unknown)
            success = True
            break
        except AssertionError:
            success = False
            
    assert success == True, "Failed to generate dataset, keep getting empty rows/columns. Tried %s times for fraction %s." % (attempts,fraction_unknown)
    
    # Store all matrices in text files
    numpy.savetxt(open(output_folder+"U.txt",'w'),U)
    numpy.savetxt(open(output_folder+"V.txt",'w'),V)
    numpy.savetxt(open(output_folder+"R_true.txt",'w'),true_R)
    numpy.savetxt(open(output_folder+"R.txt",'w'),R)
    numpy.savetxt(open(output_folder+"M.txt",'w'),M)
    
    print "Mean R: %s. Variance R: %s. Min R: %s. Max R: %s." % (numpy.mean(R),numpy.var(R),R.min(),R.max())
    fig = plt.figure()
    plt.hist(R.flatten(),bins=range(0,int(R.max())+1))
    plt.show()