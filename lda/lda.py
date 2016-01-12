import numpy as np
import scipy.sparse as sp

class lda():
    def __init__(self,n_topics=10,gibbs_samlpes_times=10,alfa=None,beta=None):
        self.n_topics = n_topics
        self.gibbs_samples_times = gibbs_samlpes_times
        self.alfa = 50.0/n_topics if alfa is None else alfa
        self.beta = 0.001 if beta is None else beta

    def fit(self,X):
        """learn model for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples,n_features)
            document word matrix

        Returns
        -------
        self
        """
        D , V = X.shape

        pass

    def get_parameters(self):
        pass