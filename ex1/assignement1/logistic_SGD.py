from cost_function import cost_function
from gradient_function import gradient_function
import numpy as np
import time


def logistic_SGD(X, y, num_iter=100000, alpha=0.01):
    """
    Perform logistic regression with stochastic gradient descent.

    Args:
        theta_0: Initial value for parameters of shape [num_features]
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train, 1]
        num_iter: Number of iterations of SGD
        alpha: The learning rate

    Returns:
        theta: The value of the parameters after logistic regression

    """
    theta = np.zeros(X.shape[1])
    losses = []
    for i in range(num_iter):
        start = time.time()
        #######################################################################
        # TODO:                                                               #
        # Perform one step of stochastic gradient descent:                    #
        #   - Select a single training example at random                      #
        #   - Update theta based on alpha and using gradient_function         #
        #                                                                     #
        #######################################################################
        #create random number
        random = int(np.floor(np.random.random() * X.shape[0]))
        #make sure dimension of variables is appropriat for further calculation
        random_X = X[random, np.newaxis]
        random_y = y[random, np.newaxis]
    
        grad = gradient_function(theta, random_X, random_y)

        # make sure dimension of variables is appropriat for further calculation
        theta = theta[:, np.newaxis]
        theta = theta + alpha * grad
        # set dimension of theta back to default
        theta = np.squeeze(theta)

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        if i % 10000 == 0:
            exec_time = time.time() - start
            loss = cost_function(theta, X, y)
            losses.append(loss)
            print('Iter {}/{}: cost = {}  ({}s)'.format(i, num_iter, loss, exec_time))
            alpha *= 0.9

    return theta, losses