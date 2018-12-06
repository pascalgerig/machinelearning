import numpy as np
import time


def kmeans(X, k, max_iter=100):
    """
    Perform k-means clusering on the data X with k number of clusters.

    Args:
        X: The data to be clustered of shape [n, num_features]
        k: The number of cluster centers to be used

    Returns:
        centers: A matrix of the computed cluster centers of shape [k, num_features]
        assign: A vector of cluster assignments for each example in X of shape [n] 

    """

    centers = None
    assign = None
    i=0
    
    start = time.time()


    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of the input data X and store the         #
    # resulting cluster-centers as well as cluster assignments.           #
    #                                                                     #
    #######################################################################

    #TODO maybe generate random number inside entire range of points instead of range 0 - 1
    centers = np.random.rand(k, X.shape[1])
    assign = np.zeros(X.shape[0])

    for i in range(max_iter):
        #loop over each sample and assign to proper cluster
        for x in range(assign.shape[0]):
            distances = np.sqrt(np.sum((centers - X[x]) ** 2, axis=1))
            assign[x] = np.argmin(distances)

        #set cluster centers to center of assigned samples
        for x in range(centers.shape[0]):
            filtered = []
            for y in range(X.shape[0]):
                if assign[y] == x:
                    #print(X[y])
                    filtered.append(X[y])
            centers[x] = np.mean(np.asarray(filtered), axis=0)


    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    
    exec_time = time.time()-start
    print('Number of iterations: {}, Execution time: {}s'.format(i+1, exec_time))
    
    return centers, assign