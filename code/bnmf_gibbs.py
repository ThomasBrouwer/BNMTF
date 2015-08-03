"""
Gibbs sampler for non-negative matrix factorisation, as introduced by Schmidt
et al. (2009).
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")
from BNMTF.code.exponential import Exponential
from BNMTF.code.gamma import Gamma
from BNMTF.code.truncated_normal import TruncatedNormal

import numpy, itertools

class bnmf_gibbs:
    def __init__(self,R,M,K):
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K
        
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)

        (self.I,self.J) = self.R.shape
        self.check_empty_rows_columns()      
        
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
    def train(self,iterations):
        self.initialise()
        self.run(iterations)


    # Initialise U, V, and tau, by drawing values from an Exp and Gamma distribution
    def initialise(self,lambdaU,lambdaV,alpha,beta):
        self.U = numpy.zeros((self.I,self.K))
        for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
            self.U[i,k] = Exponential(lambdaU[i][k]).draw()
            
        self.V = numpy.zeros((self.J,self.K))
        for j,k in itertools.product(xrange(0,self.J),xrange(0,self.K)):
            self.V[j,k] = Exponential(lambdaV[j][k]).draw()
            
        self.tau = Gamma(alpha,beta).draw()


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
                
            self.tau = Gamma(self.alpha(),self.beta())
            
            self.all_U[i], self.all_V[i], self.all_tau[i] = numpy.copy(self.U), numpy.copy(self.V), self.tau
        
        return (self.all_U, self.all_V, self.all_tau)
        
    def tauU(self,i,k):
        #TODO: return tau for Uik        
        return   
        
    def muU(self,tauUik,i,k):
        #TODO: return mu for Uik        
        return 
        
    def tauV(self,j,k):
        #TODO: return tau for Vjk        
        return   
        
    def muV(self,tauVjk,j,k):
        #TODO: return mu for Vjk        
        return 
        
    def alpha(self):
        #TODO: return alpha* for tau      
        return 
    
    def beta(self):
        #TODO: return beta* for tau      
        return 


    # Compute the expectation of U and V, and use it to predict missing values
    def predict(self):
        return