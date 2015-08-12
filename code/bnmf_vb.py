"""
Variational Bayesian inference for non-negative matrix factorisation.

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the number of latent factors
- priors = { 'alpha' = alpha_R, 'beta' = beta_R, 'lambdaU' = [[lambdaUik]], 'lambdaV' = [[lambdaVjk]] },
    a dictionary defining the priors over tau, U, V.
    
Initialisation can be done by running the initialise() function. We initialise as follows:
- muU[i,k], tauU[i,k] = lambdaU[i,k], 1
- muV[j,k], tauV[j,k] = lambdaV[j,k], 1
- alpha_s, beta_s = alpha, beta
Alternatively, you can pass a dictionary { 'muU', 'tauU', 'muV', 'tauV', 'alpha_s', 'beta_s' }

Usage of class:
    BNMF = bnmf_vb(R,M,K,priors)
    BNMF.initisalise()      (or: BNMF.initialise(values=dict))
    BNMF.run(iterations)
Or:
    BNMF = bnmf_vb(R,M,K,priors)
    BNMF.train(init,iterations)

We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMF.predict(M_pred)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
"""

from distributions.gamma import Gamma
from distributions.truncated_normal import TruncatedNormal

import numpy, itertools, math, scipy
from scipy.stats import norm

class bnmf_vb:
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
        
        assert self.lambdaU.shape == (self.I,self.K), "Prior matrix lambdaU has the wrong shape: %s instead of (%s, %s)." % (self.lambdaU.shape,self.I,self.K)
        assert self.lambdaV.shape == (self.J,self.K), "Prior matrix lambdaV has the wrong shape: %s instead of (%s, %s)." % (self.lambdaV.shape,self.J,self.K)
                   
            
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
        return self.run(iterations)


    # Initialise U, V, and tau. 
    def initialise(self,values={}):
        self.muU = values['muU'] if 'muU' in values else 1./self.lambdaU # expectation of Exp(lambdaUik)
        self.tauU = values['tauU'] if 'tauU' in values else numpy.ones((self.I,self.K))
        self.muV = values['muV'] if 'muV' in values else 1./self.lambdaV # expectation of Exp(lambdaVjk)
        self.tauV = values['tauV'] if 'tauV' in values else numpy.ones((self.J,self.K))
        self.alpha_s = values['alpha_s'] if 'alpha_s' in values else self.alpha
        self.beta_s = values['beta_s'] if 'beta_s' in values else self.beta
        
        self.expU, self.varU = numpy.zeros((self.I,self.K)), numpy.zeros((self.I,self.K))
        self.expV, self.varV = numpy.zeros((self.J,self.K)), numpy.zeros((self.J,self.K))
        
        for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
            self.update_exp_U(i,k)
        for j,k in itertools.product(xrange(0,self.J),xrange(0,self.K)):
            self.update_exp_V(j,k)
        self.update_exp_tau()


    # Run the Gibbs sampler
    def run(self,iterations):
        self.all_exp_tau = []  # to check for convergence     
        
        for it in range(0,iterations):
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
                self.update_U(i,k)
                self.update_exp_U(i,k)
                
            for j,k in itertools.product(xrange(0,self.J),xrange(0,self.K)):
                self.update_V(j,k)
                self.update_exp_V(j,k)
                
            self.update_tau()
            self.update_exp_tau()
            self.all_exp_tau.append(self.exptau)
            
            perf, elbo = self.predict(self.M), self.elbo()
            print "Iteration %s. ELBO: %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,elbo,perf['MSE'],perf['R^2'],perf['Rp'])
            
        return
        
        
    # Compute the ELBO
    def elbo(self):
        return self.size_Omega / 2. * ( self.explogtau - math.log(2*math.pi) ) \
             - self.exptau / 2. * self.exp_square_diff() \
             + numpy.log(self.lambdaU).sum() - ( self.lambdaU * self.expU ).sum() \
             + numpy.log(self.lambdaV).sum() - ( self.lambdaV * self.expV ).sum() \
             + self.alpha * math.log(self.beta) - scipy.special.gammaln(self.alpha) \
             + (self.alpha - 1.)*self.explogtau - self.beta * self.exptau \
             - self.alpha_s * math.log(self.beta_s) + scipy.special.gammaln(self.alpha_s) \
             - (self.alpha_s - 1.)*self.explogtau + self.beta_s * self.exptau \
             - .5*numpy.log(self.tauU).sum() + self.I*self.K/2.*math.log(2*math.pi) \
             + numpy.log(1. - norm.cdf(-self.muU*numpy.sqrt(self.tauU))).sum() \
             + ( self.tauU / 2. * ( self.varU + (self.expU - self.muU)**2 ) ).sum() \
             - .5*numpy.log(self.tauV).sum() + self.J*self.K/2.*math.log(2*math.pi) \
             + numpy.log(1. - norm.cdf(-self.muV*numpy.sqrt(self.tauV))).sum() \
             + ( self.tauV / 2. * ( self.varV + (self.expV - self.muV)**2 ) ).sum()
        
        
    # Update the parameters for the distributions
    def update_tau(self):   
        self.alpha_s = self.alpha + self.size_Omega/2.0
        self.beta_s = self.beta + 0.5*self.exp_square_diff()
        
    def exp_square_diff(self): # Compute: sum_Omega E_q(U,V) [ ( Rij - Ui Vj )^2 ]
        return(self.M *( ( self.R - numpy.dot(self.expU,self.expV.T) )**2 + \
                         ( numpy.dot(self.varU+self.expU**2, (self.varV+self.expV**2).T) - numpy.dot(self.expU**2,(self.expV**2).T) ) ) ).sum()
        
    def update_U(self,i,k):       
        self.tauU[i,k] = self.exptau*(self.M[i]*( self.varV[:,k] + self.expV[:,k]**2 )).sum()
        #self.tauU[i,k] = self.exptau * sum([( self.varV[j,k] + self.expV[j,k]**2 ) for j in range(0,self.J) if self.M[i,j]])
        self.muU[i,k] = 1./self.tauU[i,k] * (-self.lambdaU[i,k] + self.exptau*(self.M[i] * ( (self.R[i]-numpy.dot(self.expU[i],self.expV.T)+self.expU[i,k]*self.expV[:,k])*self.expV[:,k] )).sum()) 
        #self.muU[i,k] = 1./self.tauU[i,k] * (-self.lambdaU[i,k] + self.exptau*sum([(self.R[i,j]-numpy.dot(self.expU[i],self.expV[j].T)+self.expU[i,k]*self.expV[j,k])*self.expV[j,k] for j in range(0,self.J) if self.M[i,j]]))
        
    def update_V(self,j,k):
        self.tauV[j,k] = self.exptau*(self.M[:,j]*( self.varU[:,k] + self.expU[:,k]**2 )).sum()
        #self.tauV[j,k] = self.exptau * sum([( self.varU[i,k] + self.expU[i,k]**2 ) for i in range(0,self.I) if self.M[i,j]])
        self.muV[j,k] = 1./self.tauV[j,k] * (-self.lambdaV[j,k] + self.exptau*(self.M[:,j] * ( (self.R[:,j]-numpy.dot(self.expU,self.expV[j])+self.expU[:,k]*self.expV[j,k])*self.expU[:,k] )).sum()) 
        #self.muV[j,k] = 1./self.tauV[j,k] * (-self.lambdaV[j,k] + self.exptau*sum([(self.R[i,j]-numpy.dot(self.expU[i],self.expV[j].T)+self.expU[i,k]*self.expV[j,k])*self.expU[j,k] for i in range(0,self.I) if self.M[i,j]]))
        
    # Update the expectations and variances
    def update_exp_U(self,i,k):
        tn = TruncatedNormal(self.muU[i,k],self.tauU[i,k])
        self.expU[i,k] = tn.expectation()
        self.varU[i,k] = tn.variance()
        
    def update_exp_V(self,j,k):
        tn = TruncatedNormal(self.muV[j,k],self.tauV[j,k])
        self.expV[j,k] = tn.expectation()
        self.varV[j,k] = tn.variance()
        
    def update_exp_tau(self):
        gm = Gamma(self.alpha_s,self.beta_s)
        self.exptau = gm.expectation()
        self.explogtau = gm.expectation_log()


    # Compute the expectation of U and V, and use it to predict missing values
    def predict(self,M_pred):
        R_pred = numpy.dot(self.expU,self.expV.T)
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
        
    # Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation)
    def compute_MSE(self,M,R,R_pred):
        return (M * (R-R_pred)**2).sum() / float(M.sum())
        
    def compute_R2(self,M,R,R_pred):
        mean = (M*R).sum() / float(M.sum())
        SS_total = float((M*(R-mean)**2).sum())
        SS_res = float((M*(R-R_pred)**2).sum())
        return 1. - SS_res / SS_total
        
    def compute_Rp(self,M,R,R_pred):
        mean_real = (M*R).sum() / float(M.sum())
        mean_pred = (M*R_pred).sum() / float(M.sum())
        covariance = (M*(R-mean_real)*(R_pred-mean_pred)).sum()
        variance_real = (M*(R-mean_real)**2).sum()
        variance_pred = (M*(R_pred-mean_pred)**2).sum()
        return covariance / float(math.sqrt(variance_real)*math.sqrt(variance_pred))