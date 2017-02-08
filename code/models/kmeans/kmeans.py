import numpy, random, time

max_iterations = 200 # safeguard - if it takes more than this many iterations, stop

# Cluster the rows of the dataset X, with missing values indicated by M.
class KMeans:
    def __init__(self,X,M,K,resolve_empty='singleton'):
        self.X = numpy.copy(X)
        self.M = numpy.copy(M)
        self.K = K
        self.resolve_empty = resolve_empty
        
        assert len(self.X.shape) == 2, "Input matrix X is not a two-dimensional array, but instead %s-dimensional." % len(self.X.shape)
        assert self.X.shape == self.M.shape, "Input matrix X is not of the same size as the indicator matrix M: %s and %s respectively." % (self.X.shape,self.M.shape)
        assert self.K > 0, "K should be greater than 0."
        
        (self.no_points,self.no_coordinates) = self.X.shape
        self.no_unique_points = len(set([tuple(l) for l in self.X.tolist()]))    
        
        if self.no_points < self.K: print "Want %s clusters but only have %s datapoints!" % (self.K,self.no_points)
        if self.no_unique_points < self.K: print "Want %s clusters but only have %s unique datapoints!" % (self.K,self.no_unique_points)
        
        # Compute lists of which indices are known (from M) for each row/column
        self.omega_rows = [[j for j in range(0,self.no_coordinates) if self.M[i,j]] for i in range(0,self.no_points)]        
        self.omega_columns = [[i for i in range(0,self.no_points) if self.M[i,j]] for j in range(0,self.no_coordinates)]
                    
        # Assert none of the rows are entirely unknown values
        for i,omega_row in enumerate(self.omega_rows):
            assert len(omega_row) != 0, "Fully unobserved row in X, row %s." % i
            
        # Columns can be entirely unknown - they just don't influence the clustering - but we need to remove them
        columns_to_remove = [j for j,omega_column in enumerate(self.omega_columns) if len(omega_column) == 0]
        if len(columns_to_remove) > 0:
            print "WARNING: removed columns %s for K-means clustering as they have no observed datapoints." % columns_to_remove
            self.X = numpy.delete(self.X,columns_to_remove,axis=1)
            self.M = numpy.delete(self.M,columns_to_remove,axis=1)
            self.no_coordinates -= len(columns_to_remove)
            self.omega_rows = [[j for j in range(0,self.no_coordinates) if self.M[i,j]] for i in range(0,self.no_points)]        
            self.omega_columns = [[i for i in range(0,self.no_points) if self.M[i,j]] for j in range(0,self.no_coordinates)]
            
        # Initialise the distances from data points to the assigned cluster centroids to zeros
        self.distances = numpy.zeros(self.no_points)
    
    
    """ Initialise the cluster centroids randomly """
    def initialise(self,seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Compute the mins and maxes of the columns - i.e. the min and max of each dimension
        self.mins = [min([self.X[i,j] for i in self.omega_columns[j]]) for j in range(0,self.no_coordinates)]
        self.maxs = [max([self.X[i,j] for i in self.omega_columns[j]]) for j in range(0,self.no_coordinates)]
        
        # Randomly initialise the cluster centroids
        self.centroids = [self.random_cluster_centroid() for k in xrange(0,self.K)]
        self.cluster_assignments = numpy.array([-1 for d in xrange(0,self.no_points)])
        self.mask_centroids = numpy.ones((self.K,self.no_coordinates))


    # Randomly place a new cluster centroids, picking uniformly between the min and max of each coordinate
    def random_cluster_centroid(self):
        centroid = []
        for coordinate in xrange(0,self.no_coordinates):
            value = random.uniform(self.mins[coordinate],self.maxs[coordinate])
            centroid.append(value)     
        return centroid    
    
            
    """ Perform the clustering, until there is no change """
    def cluster(self):
        iteration = 1
        change = True
        while change:
            print "Iteration: %s." % iteration
            iteration += 1
            change = self.assignment()
            self.update()
            
            if iteration >= max_iterations:
                print "WARNING: did not converge, stopped after %s iterations." % max_iterations
                break
            
        # At the end, we create a binary matrix indicating which points were assigned to which cluster
        self.create_matrix()


    """ Assign each data point to the closest cluster, and return whether any reassignments were made """
    def assignment(self):
        self.data_point_assignments = [[] for k in xrange(0,self.K)]
        
        change = False
        for d,(data_point,mask) in enumerate(zip(self.X,self.M)):
            old_c = self.cluster_assignments[d]
            new_c = self.closest_cluster(data_point,d,mask)
            
            self.cluster_assignments[d] = new_c
            self.data_point_assignments[new_c].append(d)
            
            change = (change or old_c != new_c)
        return change
    
    
    # Compute the MSE to each of the clusters, and return the index of the closest cluster.
    # If two clusters are equally far, we return the cluster with the lowest index.
    def closest_cluster(self,data_point,index_data_point,mask_d):
        closest_index = None
        closest_MSE = None
        for c,(centroid,mask_c) in enumerate(zip(self.centroids,self.mask_centroids)):
            MSE = self.compute_MSE(data_point,centroid,mask_d,mask_c)
            if (closest_MSE is None) or (MSE is not None and MSE < closest_MSE): #either no closest centroid yet, or MSE is defined and less
                closest_MSE = MSE
                closest_index = c
        self.distances[index_data_point] = closest_MSE
        return closest_index
    
    
    # Compute the Euclidean distance between the data point and the cluster centroid.
    # If they have no known values in common, we return None (=infinite distance).
    def compute_MSE(self,x1,x2,mask1,mask2):
        mask1_mask2 = numpy.multiply(mask1,mask2)
        len_overlap = mask1_mask2.sum()
        return None if len_overlap == 0 else numpy.multiply(mask1_mask2,numpy.power(numpy.subtract(x1,x2),2)).sum() / float(len_overlap)
            
        
    """ Update the centroids to the mean of the points assigned to it. 
        If for a coordinate there are no known values, we set this cluster's mask to 0 there.
        If a cluster has no points assigned to it at all, we randomly re-initialise it. """
    def update(self):
        for c in xrange(0,self.K): 
            self.update_cluster(c)
    
    # Update for one specific cluster
    def update_cluster(self,c):
        known_coordinate_values = self.find_known_coordinate_values(c)
        
        if known_coordinate_values is None:
            # Reassign a datapoint to this cluster, as long as there are enough 
            # unique datapoints. Either furthest away (singleton) or random.
            if self.no_unique_points >= self.K:
                if self.resolve_empty == 'singleton':
                    # Find the point currently furthest away from its centroid                    
                    index_furthest_away = self.find_point_furthest_away()
                    old_cluster = self.cluster_assignments[index_furthest_away]
                    
                    # Add point to new cluster
                    self.centroids[c] = self.X[index_furthest_away]
                    self.mask_centroids[c] = self.M[index_furthest_away]
                    self.distances[index_furthest_away] = 0.0
                    self.cluster_assignments[index_furthest_away] = c
                    self.data_point_assignments[c] = [index_furthest_away]
                    
                    # Remove from old cluster and update
                    self.data_point_assignments[old_cluster].remove(index_furthest_away)
                    #print c,old_cluster
                    self.update_cluster(old_cluster)
                else:
                    # Randomly re-initialise this point
                    self.centroids[c] = self.random_cluster_centroid()
                    self.mask_centroids[c] = numpy.ones(self.no_coordinates)
        else:
            # For each coordinate set the centroid to the average, or to 0 if no values are observed
            for coordinate,coordinate_values in enumerate(known_coordinate_values):
                if len(coordinate_values) == 0:
                    new_coordinate = 0              
                    new_mask = 0
                else:
                    new_coordinate = sum(coordinate_values) / float(len(coordinate_values))
                    new_mask = 1
                
                self.centroids[c][coordinate] = new_coordinate
                self.mask_centroids[c][coordinate] = new_mask
    
    
    # For a given centroid c, construct a list of lists, each list consisting of
    # all known coordinate values of data points assigned to the centroid.
    # If no points are assigned to a cluster, return None.
    def find_known_coordinate_values(self,c):
        assigned_data_indexes = self.data_point_assignments[c]
        data_points = numpy.array([self.X[d] for d in assigned_data_indexes])
        masks = numpy.array([self.M[d] for d in assigned_data_indexes])
        
        if len(assigned_data_indexes) == 0:
            lists_known_coordinate_values = None
        else: 
            lists_known_coordinate_values = [
                [v for d,v in enumerate(data_points.T[coordinate]) if masks[d][coordinate]]
                for coordinate in xrange(0,self.no_coordinates)
            ]
            
        return lists_known_coordinate_values
        
    
    # Find data point furthest away from its current cluster centroid
    def find_point_furthest_away(self):
        data_point_index = self.distances.argmax()
        return data_point_index
        
    
    # Create a binary matrix indicating the clustering (so size [no_points x K])
    def create_matrix(self):
        self.clustering_results = numpy.zeros((self.no_points,self.K))
        for d in range(0,self.no_points):
            c = self.cluster_assignments[d]
            self.clustering_results[d][c] = 1
            
        print [sum(column) for column in self.clustering_results.T]