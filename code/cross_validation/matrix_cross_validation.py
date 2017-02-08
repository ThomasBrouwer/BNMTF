"""
General framework for performing cross validation for a matrix prediction
method. This can be used to find the best parameter values for a certain 
search space. 

We expect the following arguments:
- method, a class that performs matrix prediction, with the following functions:
    -> Constructor
       Taking in matrix X, mask matrix M, and other parameters P1,P2,...
    -> train
       Taking in a number of configuration parameters, and trains the model.
    -> predict
       Taking in the complete matrix X, a mask matrix M with 1 values where
       we with to evaluate the predictions, and returns a dictionary mapping
       performance measure names to their values.
       {'MSE','R2','Rp'} (Mean Square Error, R^2, Pearson correlation coefficient)
- X, the data matrix.
- M, a mask matrix with 1 values where entries in X are known, and 0 where they are not.
- K, the number of folds for cross-validation.
- parameter_search, a list of dictionaries from parameter names to values, 
    defining the space of our parameter search.
- train_config, the additional parameters to pass to the train function (e.g. no. of iterations).
    This should be a dictionary mapping parameter names to values 
- file_performance, the location and name of the file in which we store the performances.

For each of the parameter configurations in <parameter_search>, we split the
dataset <X> into <K> folds (considering only 1 entries in <M>), and thus form
our <K> training and test sets. Then for each we train the model using the 
parameters and training configuration <train_config>. The performances are 
stored in <file_performance>

Methods:
- Constructor - simply takes in the arguments requires
- run - no arguments, runs the cross validation and stores the results in the file
- find_best_parameters - takes in the name of the evaluation criterion (e.g. 
    'MSE'), and True if low is better (False if high is better), and returns 
    the best parameters based on that, in a tuple with all the performances.
    Also logs these findings to the file.
"""

import mask

import numpy
import json

attempts_generate_M = 1000

class MatrixCrossValidation:
    def __init__(self,method,X,M,K,parameter_search,train_config,file_performance):
        self.method = method
        self.X = numpy.array(X,dtype=float)
        self.M = numpy.array(M)
        self.K = K
        self.train_config = train_config
        self.parameter_search = parameter_search
        
        self.fout = open(file_performance,'w')
        (self.I,self.J) = self.X.shape
        assert (self.X.shape == self.M.shape), "X and M are of different shapes: %s and %s respectively." % (self.X.shape,self.M.shape)
        
        self.all_performances = {}      # Performances across all folds - mapping JSON of parameters to a dictionary from evaluation criteria to a list of performances
        self.average_performances = {}  # Average performances across folds - mapping JSON of parameters to a dictionary from evaluation criteria to average performance
        self.performances = {}          # Average performances per criterion - mapping evaluation criterion to a list of average performances (same size as parameter_search)
        
        
    # Run the cross-validation
    def run(self):
        for parameters in self.parameter_search:
            print "Trying parameters %s." % (parameters)
            
            try:
                folds_test = mask.compute_folds_attempts(I=self.I,J=self.J,no_folds=self.K,attempts=attempts_generate_M,M=self.M)
                folds_training = mask.compute_Ms(folds_test)
                
                # We need to put the parameter dict into json to hash it
                self.all_performances[self.JSON(parameters)] = {}
                for i,(train,test) in enumerate(zip(folds_training,folds_test)):
                    print "Fold %s (parameters: %s)." % (i+1,parameters)
                    performance_dict = self.run_model(train,test,parameters)
                    self.store_performances(performance_dict,parameters)
                    
                self.log(parameters)
                
            except Exception as e:
                self.fout.write("Tried parameters %s but got exception: %s. \n" % (parameters,e))
                self.fout.flush()
            
            
    # Initialises and runs the model, and returns the performance on the test set
    def run_model(self,train,test,parameters):
        model = self.method(self.X,train,**parameters)
        model.train(**self.train_config)
        return model.predict(test)
        
    # Returns the sorted json of the dictionary given
    def JSON(self,d):
        # Cannot handle numpy arrays so force all numpy arrays to be lists
        d_copy = d.copy()
        for key,val in d.iteritems():
            if isinstance(val,numpy.ndarray):
                d_copy[key] = val.tolist()
                
        return json.dumps(d_copy,sort_keys=True)            
            
    # Store the performances we get back in a dictionary from criterion name to a list of performances
    def store_performances(self,performance_dict,parameters):
        for name in performance_dict:
            if name in self.all_performances[self.JSON(parameters)]:
                self.all_performances[self.JSON(parameters)][name].append(performance_dict[name])
            else:
                self.all_performances[self.JSON(parameters)][name] = [performance_dict[name]]
              
    # Compute the average performance of the given parameters, across the K folds
    def compute_average_performances(self,parameters):
        performances = self.all_performances[self.JSON(parameters)]     
        average_performances = { name:(sum(values)/float(len(values))) for (name,values) in performances.iteritems() }
        self.average_performances[self.JSON(parameters)] = average_performances
        
        # Also store a dictionary from evaluation criterion to a list of average performances
        for (name,avr_perf) in average_performances.iteritems():
            if name in self.performances:
                self.performances[name].append(avr_perf)
            else:
                self.performances[name] = [avr_perf]
        
    # Finds the parameter values of the best performance for the specified criterion
    def find_best_parameters(self,evaluation_criterion,low_better):
        min_or_max = min if low_better else max
        self.best_performance = min_or_max(self.performances[evaluation_criterion])
        index_best = self.performances[evaluation_criterion].index(self.best_performance)
        
        self.best_parameters = self.parameter_search[index_best]
        self.best_performances_all = self.average_performances[self.JSON(self.best_parameters)]
        
        self.log_best(index_best)
        return (self.best_parameters,self.best_performance)
        
    # Logs the performances for specific parameter values
    def log(self,parameters):
        self.compute_average_performances(parameters)
        message = "Tried parameters %s. Average performances: %s. \nAll performances: %s. \n" % (parameters,self.average_performances[self.JSON(parameters)],self.all_performances[self.JSON(parameters)])
        self.fout.write(message)
        self.fout.flush()
                     
    # Logs the best performances and parameters
    def log_best(self,index_best):
        message = "Best performances: %s. Best parameters: %s. \n" % (self.best_performances_all,self.best_parameters)
        self.fout.write(message)
        self.fout.flush()