"""
Variational Bayesian inference for non-negative matrix tri-factorisation.

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the number of row clusters
- L, the number of column clusters
- priors = { 'alpha' = alpha_R, 'beta' = beta_R, 'lambdaF' = [[lambdaFik]], 'lambdaS' = [[lambdaSkl]], 'lambdaG' = [[lambdaGjl]] },
    a dictionary defining the priors over tau, F, S, G.
    
Initialisation can be done by running the initialise() function. We initialise as follows:
- muF[i,k], tauF[i,k] = lambdaF[i,k], 1
- muS[k,l], tauS[k,l] = lambdaS[k,l], 1
- muG[j,l], tauG[j,l] = lambdaG[j,l], 1
- alpha_s, beta_s = alpha, beta
Alternatively, you can pass a dictionary { 'muF', 'tauF', 'muS', 'tauS', 'muG', 'tauG', 'alpha_s', 'beta_s' }.

Usage of class:
    BNMF = bnmf_vb(R,M,K,L,priors)
    BNMF.initisalise()      (or: BNMF.initialise(values=dict))
    BNMF.run(iterations)
Or:
    BNMF = bnmf_vb(R,M,K,L,priors)
    BNMF.train(iterations)
    
We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMF.predict(M_pred)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
"""

from distributions.gamma import Gamma
from distributions.truncated_normal import TruncatedNormal

import numpy, itertools, math, scipy
from scipy.stats import norm
import matplotlib.pyplot as plt

class bnmtf_vb:
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
    def train(self,iterations):
        self.initialise()
        return self.run(iterations)


    # Initialise U, V, and tau. 
    def initialise(self,values={}):
        self.muF = values['muF'] if 'muF' in values else 1./self.lambdaF # expectation of Exp(lambdaFik)
        self.tauF = values['tauF'] if 'tauF' in values else numpy.ones((self.I,self.K))
        self.muS = values['muS'] if 'muS' in values else 1./self.lambdaS # expectation of Exp(lambdaSkl)
        self.tauS = values['tauS'] if 'tauS' in values else numpy.ones((self.K,self.L))
        self.muG = values['muG'] if 'muG' in values else 1./self.lambdaG # expectation of Exp(lambdaGjl)
        self.tauG = values['tauG'] if 'tauG' in values else numpy.ones((self.J,self.L))
        
        self.expF, self.varF = numpy.zeros((self.I,self.K)), numpy.zeros((self.I,self.K))
        self.expS, self.varS = numpy.zeros((self.K,self.L)), numpy.zeros((self.K,self.L))
        self.expG, self.varG = numpy.zeros((self.J,self.L)), numpy.zeros((self.J,self.L))
        
        for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
            self.update_exp_F(i,k)
        for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
            self.update_exp_S(k,l)
        for j,l in itertools.product(xrange(0,self.J),xrange(0,self.L)):
            self.update_exp_G(j,l)
            
        # Initialise tau using the updates
        #self.alpha_s = values['alpha_s'] if 'alpha_s' in values else self.alpha
        #self.beta_s = values['beta_s'] if 'beta_s' in values else self.beta
        self.update_tau()
        self.update_exp_tau()


    # Run the Gibbs sampler
    def run(self,iterations):
        self.all_exp_tau = []  # to check for convergence     
        
        for it in range(0,iterations):
            for k,i in itertools.product(xrange(0,self.K),xrange(0,self.I)):
                self.update_F(i,k)
                self.update_exp_F(i,k)
                
            self.update_tau()
            self.update_exp_tau()
                
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
                self.update_S(k,l)
                self.update_exp_S(k,l)
                
            self.update_tau()
            self.update_exp_tau()
                
            for l,j in itertools.product(xrange(0,self.L),xrange(0,self.J)):
                self.update_G(j,l)
                self.update_exp_G(j,l)
                
            self.update_tau()
            self.update_exp_tau()
            self.all_exp_tau.append(self.exptau)
            
            perf, elbo = self.predict(self.M), self.elbo()
            print "Iteration %s. ELBO: %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,elbo,perf['MSE'],perf['R^2'],perf['Rp'])
            
        
    # Compute the ELBO
    def elbo(self):
        return self.size_Omega / 2. * ( self.explogtau - math.log(2*math.pi) ) \
             - self.exptau / 2. * self.exp_square_diff() \
             + numpy.log(self.lambdaF).sum() - ( self.lambdaF * self.expF ).sum() \
             + numpy.log(self.lambdaS).sum() - ( self.lambdaS * self.expS ).sum() \
             + numpy.log(self.lambdaG).sum() - ( self.lambdaG * self.expG ).sum() \
             + self.alpha * math.log(self.beta) - scipy.special.gammaln(self.alpha) \
             + (self.alpha - 1.)*self.explogtau - self.beta * self.exptau \
             - self.alpha_s * math.log(self.beta_s) + scipy.special.gammaln(self.alpha_s) \
             - (self.alpha_s - 1.)*self.explogtau + self.beta_s * self.exptau \
             - .5*numpy.log(self.tauF).sum() + self.I*self.K/2.*math.log(2*math.pi) \
             + numpy.log(0.5*scipy.special.erfc(-self.muF*numpy.sqrt(self.tauF)/math.sqrt(2))).sum() \
             + ( self.tauF / 2. * ( self.varF + (self.expF - self.muF)**2 ) ).sum() \
             - .5*numpy.log(self.tauS).sum() + self.K*self.L/2.*math.log(2*math.pi) \
             + numpy.log(0.5*scipy.special.erfc(-self.muS*numpy.sqrt(self.tauS)/math.sqrt(2))).sum() \
             + ( self.tauS / 2. * ( self.varS + (self.expS - self.muS)**2 ) ).sum() \
             - .5*numpy.log(self.tauG).sum() + self.J*self.L/2.*math.log(2*math.pi) \
             + numpy.log(0.5*scipy.special.erfc(-self.muG*numpy.sqrt(self.tauG)/math.sqrt(2))).sum() \
             + ( self.tauG / 2. * ( self.varG + (self.expG - self.muG)**2 ) ).sum()
        

    # Compute the dot product of three matrices
    def triple_dot(self,M1,M2,M3):
        return numpy.dot(M1,numpy.dot(M2,M3))
        
        
    # Update the parameters for the distributions
    def update_tau(self):   
        self.alpha_s = self.alpha + self.size_Omega/2.0
        self.beta_s = self.beta + 0.5*self.exp_square_diff()
        
    def exp_square_diff(self): # Compute: sum_Omega E_q(F,S,G) [ ( Rij - Fi S Gj )^2 ]
        return (self.M*( self.R - self.triple_dot(self.expF,self.expS,self.expG.T) )**2).sum() + \
               (self.M*( self.triple_dot(self.varF+self.expF**2, self.varS+self.expS**2, (self.varG+self.expG**2).T ) - self.triple_dot(self.expF**2,self.expS**2,(self.expG**2).T) )).sum() + \
               (self.M*( numpy.dot(self.varF, ( numpy.dot(self.expS,self.expG.T)**2 - numpy.dot(self.expS**2,self.expG.T**2) ) ) )).sum() + \
               (self.M*( numpy.dot( numpy.dot(self.expF,self.expS)**2 - numpy.dot(self.expF**2,self.expS**2), self.varG.T ) )).sum()
    
    def update_F(self,i,k):  
        varSkG = numpy.dot( self.varS[k]+self.expS[k]**2 , (self.varG+self.expG**2).T ) - numpy.dot( self.expS[k]**2 , (self.expG**2).T ) # Vector of size J
        self.tauF[i,k] = self.exptau*( numpy.dot(self.M[i], varSkG + ( numpy.dot(self.expS[k],self.expG.T) )**2 ))
        
        self.muF[i,k] = 1./self.tauF[i,k] * (
            - self.lambdaF[i,k]
            + self.exptau*( numpy.dot(self.M[i], (self.R[i]-self.triple_dot(self.expF[i],self.expS,self.expG.T)+self.expF[i,k]*numpy.dot(self.expS[k],self.expG.T) ) * numpy.dot(self.expS[k],self.expG.T) ))
            - self.exptau*( numpy.dot( self.M[i], ( numpy.dot(self.varG, self.expS[k]*numpy.dot(self.expF[i],self.expS) ) - self.expF[i,k] * numpy.dot( self.expS[k]**2, self.varG.T ) ) ) )
        ) 
        # List form:
        #self.muF[i,k] = 1./self.tauF[i,k] * (-self.lambdaF[i,k] + self.exptau * sum([
        #    sum([self.expS[k,l]*self.expG[j,l] for l in range(0,self.L)]) * \
        #    (self.R[i,j] - sum([self.expF[i,kp]*self.expS[kp,l]*self.expG[j,l] for kp,l in itertools.product(xrange(0,self.K),xrange(0,self.L)) if kp != k]))
        #    - sum([
        #        self.expS[k,l] * self.varG[j,l] * sum([self.expF[i,kp] * self.expS[kp,l] for kp in range(0,self.K) if kp != k])
        #    for l in range(0,self.L)])
        #for j in range(0,self.J) if self.M[i,j]]))
    
    def update_S(self,k,l):       
        self.tauS[k,l] = self.exptau*(self.M*( numpy.outer( self.varF[:,k]+self.expF[:,k]**2 , self.varG[:,l]+self.expG[:,l]**2 ) )).sum()
        self.muS[k,l] = 1./self.tauS[k,l] * (
            - self.lambdaS[k,l] 
            + self.exptau*(self.M * ( (self.R-self.triple_dot(self.expF,self.expS,self.expG.T)+self.expS[k,l]*numpy.outer(self.expF[:,k],self.expG[:,l]) ) * numpy.outer(self.expF[:,k],self.expG[:,l]) )).sum()
            - self.exptau*(self.M * numpy.outer( self.expF[:,k] * ( numpy.dot(self.expF,self.expS[:,l]) - self.expF[:,k]*self.expS[k,l] ), self.varG[:,l] )).sum()
            - self.exptau*(self.M * numpy.outer( self.varF[:,k], self.expG[:,l]*(numpy.dot(self.expS[k],self.expG.T) - self.expS[k,l]*self.expG[:,l]) )).sum()
        ) 
        # List form
        #self.muS[k,l] = 1./self.tauS[k,l] * (-self.lambdaS[k,l] + self.exptau * sum([
        #    self.expF[i,k]*self.expG[j,l]*(self.R[i,j] - sum([self.expF[i,kp]*self.expS[kp,lp]*self.expG[j,lp] for kp,lp in itertools.product(xrange(0,self.K),xrange(0,self.L)) if (kp != k or lp != l)]))
        #    - 1./2. * self.varF[i,k] * self.expG[j,l] * sum([self.expS[k,lp] * self.expG[j,lp] for lp in range(0,self.L) if lp != l])
        #    - 1./2. * self.expF[i,k] * self.varG[j,l] * sum([self.expF[i,kp] * self.expS[kp,l] for kp in range(0,self.K) if kp != k])
        #for i,j in itertools.product(xrange(0,self.I),xrange(0,self.J)) if self.M[i,j]]))
        
            
    def update_G(self,j,l):  
        varFSl = numpy.dot( self.varF+self.expF**2 , self.varS[:,l]+self.expS[:,l]**2 ) - numpy.dot( self.expF**2 , self.expS[:,l]**2 ) # Vector of size I
        self.tauG[j,l] = self.exptau*(numpy.dot(self.M[:,j], varFSl + ( numpy.dot(self.expF,self.expS[:,l]) )**2 ))
        
        self.muG[j,l] = 1./self.tauG[j,l] * (
            - self.lambdaG[j,l] 
            + self.exptau * numpy.dot(self.M[:,j], (self.R[:,j]-self.triple_dot(self.expF,self.expS,self.expG[j].T)+self.expG[j,l]*numpy.dot(self.expF,self.expS[:,l]) ) * numpy.dot(self.expF,self.expS[:,l]) )
            - self.exptau * numpy.dot(self.M[:,j], numpy.dot(self.varF, self.expS[:,l]*numpy.dot(self.expS,self.expG[j])) - self.expG[j,l]*numpy.dot(self.varF,self.expS[:,l]**2) )
        )
        # List form:
        #self.muG[j,l] = 1./self.tauG[j,l] * (-self.lambdaG[j,l] + self.exptau * sum([
        #    sum([self.expF[i,k]*self.expS[k,l] for k in range(0,self.K)]) * \
        #    (self.R[i,j] - sum([self.expF[i,k]*self.expS[k,lp]*self.expG[j,lp] for k,lp in itertools.product(xrange(0,self.K),xrange(0,self.L)) if lp != l]))
        #    - sum([
        #        self.varF[i,k] * self.expS[k,l] * sum([self.expS[k,lp] * self.expG[j,lp] for lp in range(0,self.L) if lp != l])
        #    for k in range(0,self.K)])
        #for i in range(0,self.I) if self.M[i,j]]))
        

    # Update the expectations and variances
    def update_exp_F(self,i,k):
        tn = TruncatedNormal(self.muF[i,k],self.tauF[i,k])
        self.expF[i,k] = tn.expectation()
        self.varF[i,k] = tn.variance()
        
    def update_exp_S(self,k,l):
        tn = TruncatedNormal(self.muS[k,l],self.tauS[k,l])
        self.expS[k,l] = tn.expectation()
        self.varS[k,l] = tn.variance()
        
    def update_exp_G(self,j,l):
        tn = TruncatedNormal(self.muG[j,l],self.tauG[j,l])
        self.expG[j,l] = tn.expectation()
        self.varG[j,l] = tn.variance()
        
    def update_exp_tau(self):
        gm = Gamma(self.alpha_s,self.beta_s)
        self.exptau = gm.expectation()
        self.explogtau = gm.expectation_log()


    # Compute the expectation of U and V, and use it to predict missing values
    def predict(self,M_pred):
        R_pred = self.triple_dot(self.expF,self.expS,self.expG.T)
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