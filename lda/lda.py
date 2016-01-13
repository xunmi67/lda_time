import numpy as np
import scipy.sparse as sp
import random
class lda():
    MAX_LEN_OF_DOC = 1000
    def __init__(self,n_topics=10,alfa=None,beta=None):
        self.n_topics = n_topics
        self.alfa = 50.0/n_topics if alfa is None else alfa
        self.beta = 0.001 if beta is None else beta

    def fit(self,X):
        """learn model for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape[0] = (n words),words of a doc

        Returns
        -------
        self
        """
        X = X if sp.isspmatrix_csr(X) else sp.csr_matrix(X)
        D  = X.shape[0]
        V = X.max()+1
        z_len = X.sum(1).tonarray()[:,0]
        z_vectors = [[random.randint(0,self.n_topics-1) for i in range(self.n_topics)] for m in range(D)]
        k_v_mat = sp.csr_matrix(np.zeros((self.n_topics,V),np.int32))
        m_k_mat = sp.csr_matrix(np.zeros((D,self.n_topics),np.int32))
        for d in range(D):
            indices = X.indices[X.indptr[d],X.indptr[d+1]]
            z_d = z_vectors[d]
            assert len(indices) == len(z_d)
            for i in range(len(indices)):
                w_i = indices(w_i)
                z_i = z_d[i]
                k_v_mat[z_i][w_i] += 1
                m_k_mat[d][z_i] += 1

        # gibbs sample,burn in process
        for sample_time in range(self.n_topics*self.MAX_LEN_OF_DOC*20):
            for d in D:
                w_d = X.indices[X.indptr[d],X.indptr[d+1]]
                z_d = z_vectors[d]
                for i in range(w_d):
                    k_v_mat[z_d[i]][w_d[i]] -= 1
                    m_k_mat[d][z_d[i]] -= 1
                    # cal sample probability Q
                    Q = []
                    z_i = i
                    k_v_mat[z_i][w_d[i]] += 1
                    m_k_mat[d][z_i] += 1




    def get_parameters(self):
        pass