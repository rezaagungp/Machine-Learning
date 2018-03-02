import numpy as np

class AdalineGD(object):
    """Adaptive Lineara Neuron Classifier

    parameters
    ----------
    eta : float => Learning rate (0.0 - 1.0)
    n_iter : int => eposch for training dataset

    attributes
    ----------
    w_ : array-1D => weights for each perceptron
    errors_ : list => error score for each epoch
    """

    def __init__(self, eta = 0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter

    
    def fit(self, X, y):
        """ fit training data.
        
        Parameters
        ----------
        x : {array-like}, shape = [n_sample, n_feature]
            n_samples is number of total data in dataset
            n_features is number of feature or attributes
        y : array-like, shape = [n_sample]
            the label for each data

        Returns
        -------
        self : object
        
        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        
        return self
    

    def net_input(self, X):
        """calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)
    

    def predict(self, X):
        """Return label after each epoch"""
        return np.where(self.activation(X) >= 0.0, 1, -1)