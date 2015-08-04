"""
Gibbs sampler for non-negative matrix factorisation, as introduced by Schmidt
et al. (2009).

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the number of latent factors
- priors = { 'alpha' = alpha_R, 'beta' = beta_R, 'lambdaU' = [[lambdaUik]], 'lambdaV' = [[lambdaVjk]] },
    a dictionary defining the priors over tau, U, V.
    
Initialisation can be done by running the initialise() function, with argument init:
- init='random' -> draw initial values randomly from priors Exp, Gamma
- init='exp'    -> use the expectation of the priors Exp, Gamma
Alternatively, you can define your own initial values for U, V, and tau.

Usage of class:
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.initisalise(init)
    BNMF.run(iterations)
Or:
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.train(init,iterations)
    
This returns a tuple (Us,Vs,taus) of lists of U, V, tau values - of size <iterations>.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")
from BNMTF.code.distributions.exponential import Exponential
from BNMTF.code.distributions.gamma import Gamma
from BNMTF.code.distributions.truncated_normal import TruncatedNormal

import numpy, itertools

class bnmf_gibbs:
    def __init__(self,R,M,K,priors):
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K
        
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)
            
        (self.I,self.J) = self.R.shape
        self.size_Omega = self.M.sum()
        self.check_empty_rows_columns()      
        
        self.alpha, self.beta, self.lambdaU, self.lambdaV = \
            float(priors['alpha']), float(priors['beta']), numpy.array(priors['lambdaU']), numpy.array(priors['lambdaV'])
        
        assert len(self.lambdaU.shape) == 2, "Prior matrix lambdaU is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.lambdaU.shape)
        assert len(self.lambdaV.shape) == 2, "Prior matrix lambdaV is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.lambdaV.shape)
            
        
    # Raise an exception if an entire row or column is empty
    def check_empty_rows_columns(self):
        sums_columns = self.M.sum(axis=0)
        sums_rows = self.M.sum(axis=1)
                    
        # Assert none of the rows or columns are entirely unknown values
        for i,c in enumerate(sums_rows):
            assert c != 0, "Fully unobserved row in R, row %s." % i
        for j,c in enumerate(sums_columns):
            assert c != 0, "Fully unobserved column in R, column %s." % j


    # Initialise and run the sampler
    def train(self,init,iterations):
        self.initialise(init=init)
        return self.run(iterations)


    # Initialise U, V, and tau. If init='random', draw values from an Exp and Gamma distribution. If init='exp', set it to the expectation values.
    def initialise(self,init='random'):
        assert init in ['random','exp'], "Unknown initialisation option: %s. Should be 'random' or 'exp'." % init
        self.U = numpy.zeros((self.I,self.K))
        self.V = numpy.zeros((self.J,self.K))
        
        if init == 'random':
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
                self.U[i,k] = Exponential(self.lambdaU[i][k]).draw()
            for j,k in itertools.product(xrange(0,self.J),xrange(0,self.K)):
                self.V[j,k] = Exponential(self.lambdaV[j][k]).draw()
            self.tau = Gamma(self.alpha,self.beta).draw()
            
        elif init == 'exp':
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
                self.U[i,k] = 1.0/self.lambdaU[i][k]
            for j,k in itertools.product(xrange(0,self.J),xrange(0,self.K)):
                self.V[j,k] = 1.0/self.lambdaV[j][k]
            self.tau = self.alpha/self.beta


    # Run the Gibbs sampler
    def run(self,iterations):
        self.all_U = numpy.zeros((iterations,self.I,self.K))  
        self.all_V = numpy.zeros((iterations,self.J,self.K))   
        self.all_tau = numpy.zeros(iterations)
        
        for i in range(1,iterations+1):
            print "Iteration %s." % i
            
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
                tauUik = self.tauU(i,k)
                muUik = self.muU(tauUik,i,k)
                self.U[i,k] = TruncatedNormal(muUik,tauUik).draw()
                
            for j,k in itertools.product(xrange(0,self.J),xrange(0,self.K)):
                tauVjk = self.tauV(j,k)
                muVjk = self.muU(tauVjk,j,k)
                self.U[i,k] = TruncatedNormal(muVjk,tauVjk).draw()
                
            self.tau = Gamma(self.alpha_s(),self.beta_s())
            
            self.all_U[i], self.all_V[i], self.all_tau[i] = numpy.copy(self.U), numpy.copy(self.V), self.tau
        
        return (self.all_U, self.all_V, self.all_tau)
        
        
    def alpha_s(self):   
        return self.alpha + self.size_Omega/2.0
    
    def beta_s(self):   
        return self.beta + 0.5*(self.M*(self.R-numpy.dot(self.U,self.V.T))**2).sum()
        
    def tauU(self,i,k):       
        return self.tau*(self.M[i]*self.V[:,k]**2).sum()
        
    def muU(self,tauUik,i,k):
        return 1./tauUik * (-self.lambdaU[i,k] + self.tau*(self.M[i] * ( (self.R[i]-numpy.dot(self.U[i],self.V.T)+self.U[i,k]*self.V[:,k])*self.V[:,k] )).sum()) 
        
    def tauV(self,j,k):
        return self.tau*(self.M[:,j]*self.U[:,k]**2).sum()
        
    def muV(self,tauVjk,j,k):
        return 1./tauVjk * (-self.lambdaV[j,k] + self.tau*(self.M[:,j] * ( (self.R[:,j]-numpy.dot(self.U,self.V[j])+self.U[:,k]*self.V[j,k])*self.U[:,k] )).sum()) 


    # Return the average value for U, V, tau. Throw away the first <burn_in> samples, and then use every <thinning>th after.
    def approx_expectation(self,burn_in,thinning):
        indices = range(burn_in,len(self.all_U),thinning)
        Us, Vs, taus = [self.all_U[i] for i in indices], [self.all_V[i] for i in indices], [self.all_tau[i] for i in indices]
        return (Us, Vs, taus)


    # Compute the expectation of U and V, and use it to predict missing values
    def predict(self):
        return