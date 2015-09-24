"""
Class for doing model selection for BNMF, minimising the BIC, AIC, or MSE.

We expect the following arguments:
- classifier    - a class for BNMF, with methods: 
                    __init__(R,M,K,priors), 
                    initialise(initUV), 
                    run(iterations), 
                    quality(metric)         - metric in ['AIC','BIC','loglikelihood','MSE']
                    or quality(metric,burn_in,thinning) for Gibbs
- values_K      - a list of values for K
- R             - the dataset
- M             - the mask matrix
- prior         - the prior values for BNMF. This should be a dictionary of the form:
                    { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
                  where lambdaU and lambdaV are a single value.
- initUV        - the initialisation of U and V - either 'exp' or 'random'
- iterations    - number of iterations to run 

The line search can be started by running search().
If we use Gibbs then we run search(burn_in,thinning).

After that, the values for each metric ('BIC','AIC','loglikelihood','MSE') can 
be obtained using all_values(metric), and the best value of K can be returned
using best_value(metric).

We use the optimised Variational Bayes algorithm for BNMF.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

import numpy

metrics = ['BIC','AIC','loglikelihood','MSE']

class LineSearch:
    def __init__(self,classifier,values_K,R,M,priors,initUV,iterations):
        self.classifier = classifier
        self.values_K = values_K
        self.R = R
        self.M = M
        (self.I,self.J) = self.R.shape
        self.priors = priors
        self.initUV = initUV
        self.iterations = iterations
        
        self.all_performances = {
            metric : []
            for metric in metrics
        }
    
    
    def search(self,burn_in=None,thinning=None):
        for K in self.values_K:
            print "Running line search for BNMF. Trying K = %s." % K
                        
            priors = self.priors.copy()
            priors['lambdaU'] = self.priors['lambdaU']*numpy.ones((self.I,K))
            priors['lambdaV'] = self.priors['lambdaV']*numpy.ones((self.J,K))
            
            BNMF = self.classifier(self.R,self.M,K,priors)
            BNMF.initialise(init=self.initUV)
            BNMF.run(iterations=self.iterations)
            
            for metric in metrics:
                if burn_in is not None and thinning is not None:
                    quality = BNMF.quality(metric,burn_in,thinning)
                else:
                    quality = BNMF.quality(metric)
                self.all_performances[metric].append(quality)
        
        print "Finished running line search for BNMF."
    
    
    def all_values(self,metric):
        assert metric in metrics, "Unrecognised metric name: %s." % metric
        return self.all_performances[metric]
    
    
    def best_value(self,metric):
        assert metric in metrics, "Unrecognised metric name: %s." % metric
        return self.values_K[self.all_values(metric).index(min(self.all_values(metric)))]