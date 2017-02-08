"""
Algorithm for running cross validation with line search on a dataset, for the
probabilistic NMF models (VB, Gibbs, ICM).

Arguments:
- classifier        - the classifier we train
- R                 - the dataset
- M                 - mask matrix
- values_K          - list specifying search range for K
- folds             - number of folds
- priors            - dictionary from hyperparameter names to values
- init_UV           - how we initialise U and V
- iterations        - number of iterations we run each model
- restarts          - the number of times we try each model when doing model selection
- quality_metric    - the metric we use to measure model quality - MSE, AIC, or BIC
- file_performance  - the file in which we store the performances

We start the search using run(). If we use ICM we use run(minimum_TN=<>)
run(burn_in=<>,thinning=<>).
"""

import mask
from line_search_bnmf import LineSearch

import numpy

metrics = ['MSE','AIC','BIC']
measures = ['R^2','MSE','Rp']
attempts_generate_M = 100 

class LineSearchCrossValidation:
    def __init__(self,classifier,R,M,values_K,folds,priors,init_UV,iterations,restarts,quality_metric,file_performance):
        self.classifier = classifier
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M)
        self.values_K = values_K
        self.folds = folds
        self.priors = priors
        self.init_UV = init_UV
        self.iterations = iterations
        self.restarts = restarts
        self.quality_metric = quality_metric
        
        self.fout = open(file_performance,'w')
        (self.I,self.J) = self.R.shape
        assert (self.R.shape == self.M.shape), "R and M are of different shapes: %s and %s respectively." % (self.R.shape,self.M.shape)
        
        assert self.quality_metric in metrics       
        
        self.performances = {}          # Performances across folds
        
        
    # Run the cross-validation
    def run(self,burn_in=None,thinning=None,minimum_TN=None):
        folds_test = mask.compute_folds_attempts(I=self.I,J=self.J,no_folds=self.folds,attempts=attempts_generate_M,M=self.M)
        folds_training = mask.compute_Ms(folds_test)

        performances_test = {measure:[] for measure in measures}
        for i,(train,test) in enumerate(zip(folds_training,folds_test)):
            print "Fold %s." % (i+1)
            
            # Run the line search
            line_search = LineSearch(
                classifier=self.classifier,
                values_K=self.values_K,
                R=self.R,
                M=train,
                priors=self.priors,
                initUV=self.init_UV,
                iterations=self.iterations,
                restarts=self.restarts)
            line_search.search(burn_in=burn_in,thinning=thinning,minimum_TN=minimum_TN)
            
            # Store the model fits, and find the best one according to the metric    
            all_performances = line_search.all_values(metric=self.quality_metric)
            self.fout.write("All model fits for fold %s, metric %s: %s.\n" % (i+1,self.quality_metric,all_performances)) 
            self.fout.flush()
            
            best_K = line_search.best_value(metric=self.quality_metric)
            self.fout.write("Best K for fold %s: %s.\n" % (i+1,best_K))
            
            # Train a model with this K and measure performance on the test set
            performance = self.run_model(train,test,best_K,burn_in=burn_in,thinning=thinning,minimum_TN=minimum_TN)
            self.fout.write("Performance: %s.\n\n" % performance)
            self.fout.flush()
            
            for measure in measures:
                performances_test[measure].append(performance[measure])
        
        # Store the final performances and average
        average_performance_test = self.compute_average_performance(performances_test)
        message = "Average performance: %s. \nPerformances test: %s." % (average_performance_test,performances_test)
        print message
        self.fout.write(message)        
        self.fout.flush()
        
              
    # Compute the average performance of the given list of performances (MSE, R^2, Rp)
    def compute_average_performance(self,performances):
        return { measure:(sum(values)/float(len(values))) for measure,values in performances.iteritems() }

        
    # Initialises and runs the model, and returns the performance on the test set
    def run_model(self,train,test,K,burn_in=None,thinning=None,minimum_TN=None):
        # We train <restarts> models, and use the one with the best log likelihood to make predictions   
        best_loglikelihood = None
        best_performance = None
        for r in range(0,self.restarts):
            model = self.classifier(
                R=self.R,
                M=train,
                K=K,
                priors=self.priors
            )
            model.initialise(self.init_UV)
            
            if minimum_TN is None:
                model.run(self.iterations)
            else:
                model.run(self.iterations,minimum_TN=minimum_TN)
                
            if burn_in is None or thinning is None:
                new_loglikelihood = model.quality('loglikelihood')
                performance = model.predict(test)
            else:
                new_loglikelihood = model.quality('loglikelihood',burn_in,thinning)
                performance = model.predict(test,burn_in,thinning)
                
            if best_loglikelihood is None or new_loglikelihood > best_loglikelihood:
                best_loglikelihood = new_loglikelihood
                best_performance = performance
                
            print "Trained final model, attempt %s. Log likelihood: %s." % (r+1,new_loglikelihood)            
            
        print "Best log likelihood: %s." % best_loglikelihood
        return best_performance