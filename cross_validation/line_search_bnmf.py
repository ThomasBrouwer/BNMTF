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
- restarts      - we run the classifier this many times and use the one with 
                  the highest log likelihood

The line search can be started by running search().
If we use Gibbs then we run search(burn_in=<>,thinning=<>).
If we use ICM then we use run_search(minimum_TN=<>)

After that, the values for each metric ('BIC','AIC','loglikelihood','MSE') can 
be obtained using all_values(metric), and the best value of K can be returned
using best_value(metric).
"""

metrics = ['BIC','AIC','loglikelihood','MSE','ELBO']

class LineSearch:
    def __init__(self,classifier,values_K,R,M,priors,initUV,iterations,restarts=1):
        self.classifier = classifier
        self.values_K = values_K
        self.R = R
        self.M = M
        (self.I,self.J) = self.R.shape
        self.priors = priors
        self.initUV = initUV
        self.iterations = iterations
        self.restarts = restarts
        assert self.restarts > 0, "Need at least 1 restart."
        
        self.all_performances = {
            metric : []
            for metric in metrics
        }
    
    
    def search(self,burn_in=None,thinning=None,minimum_TN=None):
        for K in self.values_K:
            print "Running line search for BNMF. Trying K = %s." % K
            best_BNMF = None
            for r in range(0,self.restarts):
                print "Restart %s for K = %s." % (r+1,K)
                BNMF = self.classifier(self.R,self.M,K,self.priors)
                BNMF.initialise(init=self.initUV)
                if minimum_TN is None:
                    BNMF.run(iterations=self.iterations)
                else:
                    BNMF.run(iterations=self.iterations,minimum_TN=minimum_TN)
                
                args = {'metric':'loglikelihood'}
                if burn_in is not None and thinning is not None:
                    args['burn_in'], args['thinning'] = burn_in, thinning
                
                if best_BNMF is None or BNMF.quality(**args) > best_BNMF.quality(**args):
                    best_BNMF = BNMF
            
            for metric in metrics:
                if burn_in is not None and thinning is not None:
                    quality = best_BNMF.quality(metric,burn_in,thinning)
                else:
                    quality = best_BNMF.quality(metric)
                self.all_performances[metric].append(quality)
        
        print "Finished running line search for BNMF."
    
    
    def all_values(self,metric):
        assert metric in metrics, "Unrecognised metric name: %s." % metric
        return self.all_performances[metric]
    
    
    def best_value(self,metric):
        assert metric in metrics, "Unrecognised metric name: %s." % metric
        return self.values_K[self.all_values(metric).index(min(self.all_values(metric)))]