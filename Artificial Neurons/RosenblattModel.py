import numpy as np


class Perceptron(object):
    """ Perceptron classifier

    Parameters
    ----------
    eta : float => Learing rate (0.0 - 1.0)
    n_iter : int => epoch for training dataset


    Attributes
    ----------
    w_ : array-1D => weights for each perceptron
    errors_ : list => error score for each epoch

    """

    def __init__(self, eta=0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
    

    def fit(self, X, y):
        """learning from training data

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            n_samples is number of total data in dataset
            n_features is number of feature or attributes
        y : array-like, shape = [n_sample]
            the label for each data

        Returns
        -------
        self : object
        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    

    def net_input(self, X):
        """Calculate Net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    

    def predict(self, X):
        """Return label data for each epoch"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)