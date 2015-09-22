"""
Gibbs sampler for non-negative matrix tri-factorisation.

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the number of row clusters
- L, the number of column clusters
- priors = { 'alpha' = alpha_R, 'beta' = beta_R, 'lambdaF' = [[lambdaFik]], 'lambdaS' = [[lambdaSkl]], 'lambdaG' = [[lambdaGjl]] },
    a dictionary defining the priors over tau, F, S, G.
    
Initialisation can be done by running the initialise() function, with argument init:
- init='random' -> draw initial values randomly from priors Exp, Gamma
- init='exp'    -> use the expectation of the priors Exp, Gamma
Alternatively, you can define your own initial values for F, S, G, and tau.

Usage of class:
    BNMF = bnmf_gibbs(R,M,K,L,priors)
    BNMF.initisalise(init)
    BNMF.run(iterations)
Or:
    BNMF = bnmf_gibbs(R,M,K,L,priors)
    BNMF.train(init,iterations)
    
This returns a tuple (Fs,Ss,Gs,taus) of lists of F, S, G, tau values - of size <iterations>.

The expectation can be computed by specifying a burn-in and thinning rate, and using:
    BNMF.approx_expectation(burn_in,thinning)

We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMF.predict(M_pred,burn_in,thinning)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
"""

from distributions.exponential import Exponential
from distributions.gamma import Gamma
from distributions.truncated_normal import TruncatedNormal

import numpy, itertools, math, time

class bnmtf_gibbs:
    def __init__(self,R,M,K,L,priors):
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K
        self.L = L
        
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)
            
        (self.I,self.J) = self.R.shape
        self.size_Omega = self.M.sum()
        self.check_empty_rows_columns()      
        
        self.alpha, self.beta, self.lambdaF, self.lambdaS, self.lambdaG = \
            float(priors['alpha']), float(priors['beta']), numpy.array(priors['lambdaF']), numpy.array(priors['lambdaS']), numpy.array(priors['lambdaG'])
        
        assert self.lambdaF.shape == (self.I,self.K), "Prior matrix lambdaF has the wrong shape: %s instead of (%s, %s)." % (self.lambdaF.shape,self.I,self.K)
        assert self.lambdaS.shape == (self.K,self.L), "Prior matrix lambdaS has the wrong shape: %s instead of (%s, %s)." % (self.lambdaS.shape,self.K,self.L)
        assert self.lambdaG.shape == (self.J,self.L), "Prior matrix lambdaG has the wrong shape: %s instead of (%s, %s)." % (self.lambdaG.shape,self.J,self.L)
            
        
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
        self.F = numpy.zeros((self.I,self.K))
        self.S = numpy.zeros((self.K,self.L))
        self.G = numpy.zeros((self.J,self.L))
        
        if init == 'random':
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
                self.F[i,k] = Exponential(self.lambdaF[i][k]).draw()
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
                self.S[k,l] = Exponential(self.lambdaS[k][l]).draw()
            for j,l in itertools.product(xrange(0,self.J),xrange(0,self.L)):
                self.G[j,l] = Exponential(self.lambdaG[j][l]).draw()
            self.tau = Gamma(self.alpha,self.beta).draw()
            
        elif init == 'exp':
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
                self.F[i,k] = 1.0/self.lambdaF[i][k]
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
                self.S[k,l] = 1.0/self.lambdaS[k][l]
            for j,l in itertools.product(xrange(0,self.J),xrange(0,self.L)):
                self.G[j,l] = 1.0/self.lambdaG[j][l]
            self.tau = self.alpha/self.beta


    # Run the Gibbs sampler
    def run(self,iterations):
        self.all_F = numpy.zeros((iterations,self.I,self.K))  
        self.all_S = numpy.zeros((iterations,self.K,self.L))   
        self.all_G = numpy.zeros((iterations,self.J,self.L))  
        self.all_tau = numpy.zeros(iterations)
        
        for it in range(0,iterations):
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
                tauFik = self.tauF(i,k)
                muFik = self.muF(tauFik,i,k)
                self.F[i,k] = TruncatedNormal(muFik,tauFik).draw()
                
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
                tauSkl = self.tauS(k,l)
                muSkl = self.muS(tauSkl,k,l)
                self.S[k,l] = TruncatedNormal(muSkl,tauSkl).draw()
                
            for j,l in itertools.product(xrange(0,self.J),xrange(0,self.L)):
                tauGjl = self.tauG(j,l)
                muGjl = self.muG(tauGjl,j,l)
                self.G[j,l] = TruncatedNormal(muGjl,tauGjl).draw()
                
            self.tau = Gamma(self.alpha_s(),self.beta_s()).draw()
            
            self.all_F[it], self.all_S[it], self.all_G[it], self.all_tau[it] = numpy.copy(self.F), numpy.copy(self.S), numpy.copy(self.G), self.tau
            perf = self.predict_while_running()
            print "Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp'])
        
        return (self.all_F, self.all_S, self.all_G, self.all_tau)
        

    # Compute the dot product of three matrices
    def triple_dot(self,M1,M2,M3):
        return numpy.dot(M1,numpy.dot(M2,M3))
        
        
    # Compute the parameters for the distributions we sample from
    def alpha_s(self):   
        return self.alpha + self.size_Omega/2.0
    
    def beta_s(self):   
        return self.beta + 0.5*(self.M*(self.R-self.triple_dot(self.F,self.S,self.G.T))**2).sum()
        
    def tauF(self,i,k):       
        return self.tau * ( self.M[i] * numpy.dot(self.S[k],self.G.T)**2 ).sum()
        
    def muF(self,tauFik,i,k):
        return 1./tauFik * (-self.lambdaF[i,k] + self.tau*(self.M[i] * ( (self.R[i]-self.triple_dot(self.F[i],self.S,self.G.T)+self.F[i,k]*numpy.dot(self.S[k],self.G.T))*numpy.dot(self.S[k],self.G.T) )).sum()) 
        
    def tauS(self,k,l):       
        return self.tau * ( self.M * numpy.outer(self.F[:,k]**2,self.G[:,l]**2) ).sum()
        
    def muS(self,tauSkl,k,l):
        return 1./tauSkl * (-self.lambdaS[k,l] + self.tau*(self.M * ( (self.R-self.triple_dot(self.F,self.S,self.G.T)+self.S[k,l]*numpy.outer(self.F[:,k],self.G[:,l]))*numpy.outer(self.F[:,k],self.G[:,l]) )).sum()) 
        
    def tauG(self,j,l):       
        return self.tau * ( self.M[:,j] * numpy.dot(self.F,self.S[:,l])**2 ).sum()
        
    def muG(self,tauGjl,j,l):
        return 1./tauGjl * (-self.lambdaG[j,l] + self.tau*(self.M[:,j] * ( (self.R[:,j]-self.triple_dot(self.F,self.S,self.G[j])+self.G[j,l]*numpy.dot(self.F,self.S[:,l]))*numpy.dot(self.F,self.S[:,l]) )).sum()) 
        

    # Return the average value for U, V, tau - i.e. our approximation to the expectations. 
    # Throw away the first <burn_in> samples, and then use every <thinning>th after.
    def approx_expectation(self,burn_in,thinning):
        indices = range(burn_in,len(self.all_F),thinning)
        exp_F = numpy.array([self.all_F[i] for i in indices]).sum(axis=0) / float(len(indices))      
        exp_S = numpy.array([self.all_S[i] for i in indices]).sum(axis=0) / float(len(indices))   
        exp_G = numpy.array([self.all_G[i] for i in indices]).sum(axis=0) / float(len(indices))  
        exp_tau = sum([self.all_tau[i] for i in indices]) / float(len(indices))
        return (exp_F, exp_S, exp_G, exp_tau)


    # Compute the expectation of U and V, and use it to predict missing values
    def predict(self,M_pred,burn_in,thinning):
        (exp_F,exp_S,exp_G,_) = self.approx_expectation(burn_in,thinning)
        R_pred = self.triple_dot(exp_F,exp_S,exp_G.T)
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
    def predict_while_running(self):
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
        MSE = self.compute_MSE(self.M,self.R,R_pred)
        R2 = self.compute_R2(self.M,self.R,R_pred)    
        Rp = self.compute_Rp(self.M,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
        
    # Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation)
    def compute_MSE(self,M,R,R_pred):
        return (M * (R-R_pred)**2).sum() / float(M.sum())
        
    def compute_R2(self,M,R,R_pred):
        mean = (M*R).sum() / float(M.sum())
        SS_total = float((M*(R-mean)**2).sum())
        SS_res = float((M*(R-R_pred)**2).sum())
        return 1. - SS_res / SS_total if SS_total != 0. else numpy.inf
        
    def compute_Rp(self,M,R,R_pred):
        mean_real = (M*R).sum() / float(M.sum())
        mean_pred = (M*R_pred).sum() / float(M.sum())
        covariance = (M*(R-mean_real)*(R_pred-mean_pred)).sum()
        variance_real = (M*(R-mean_real)**2).sum()
        variance_pred = (M*(R_pred-mean_pred)**2).sum()
        return covariance / float(math.sqrt(variance_real)*math.sqrt(variance_pred))
        
        
    # Functions for model selection, measuring the goodness of fit vs model complexity
    def quality(self,metric,burn_in,thinning):
        assert metric in ['loglikelihood','BIC','AIC','MSE'], 'Unrecognised metric for model quality: %s.' % metric
        
        (expF,expS,expG,exptau) = self.approx_expectation(burn_in,thinning)
        log_likelihood = self.log_likelihood(expF,expS,expG,exptau)
        
        if metric == 'loglikelihood':
            return log_likelihood
        elif metric == 'BIC':
            # -2*loglikelihood + (no. free parameters * log(no data points))
            return log_likelihood - 0.5 * (self.I*self.K+self.K*self.L+self.J*self.L) * math.log(self.size_Omega)
        elif metric == 'AIC':
            # -2*loglikelihood + 2*no. free parameters
            return log_likelihood - (self.I*self.K+self.K*self.L+self.J*self.L)
        elif metric == 'MSE':
            R_pred = self.triple_dot(expF,expS,expG.T)
            return self.compute_MSE(self.M,self.R,R_pred)
        
    def log_likelihood(self,expF,expS,expG,exptau):
        # Return the likelihood of the data given the trained model's parameters
        explogtau = math.log(exptau)
        return self.size_Omega / 2. * ( explogtau - math.log(2*math.pi) ) \
             - exptau / 2. * (self.M*( self.R - self.triple_dot(expF,expS,expG.T) )**2).sum()