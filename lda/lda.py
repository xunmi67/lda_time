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
#        z_len = X.sum(1).getA()[:,0]
        z_len = [X.indptr[i+1]-X.indptr[i] for i in range(D)]
        z_vectors = [[random.randint(0,self.n_topics-1) for i in range(z_len[m])] for m in range(D)]
        k_v_mat = sp.csr_matrix(np.zeros((self.n_topics,V),np.int32))
        m_k_mat = sp.csr_matrix(np.zeros((D,self.n_topics),np.int32))
        for d in range(D):
            d_w = X.data[X.indptr[d]:X.indptr[d+1]]
            z_d = z_vectors[d]
            assert len(d_w) == len(z_d)
            for i in range(len(d_w)):
                w_i = X.data[i]
                z_i = z_d[i]
                k_v_mat[z_i][w_i] += 1
                m_k_mat[d][z_i] += 1

        # gibbs sample,burn in process
        for sample_time in range(self.n_topics*self.MAX_LEN_OF_DOC*20):
            for d in D:
                w_d = X.data[X.indptr[d]:X.indptr[d+1]]
                z_d = z_vectors[d]
                for i in range(w_d):
                    k_v_mat[z_d[i]][w_d[i]] -= 1
                    m_k_mat[d][z_d[i]] -= 1
                    # cal sample probability Q
                    Q = []
                    z_i = i
                    k_v_mat[z_i][w_d[i]] += 1
                    m_k_mat[d][z_i] += 1

        # it should come to stable state,state sample, and calculate theta and phi
        self.phi = np.zeros((self.n_topics,V))
        self.theta = np.zeros((D,self.n_topics))
        stable_times = self.n_topics*self.MAX_LEN_OF_DOC*20
        for sample_time in range(stable_times):
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
            self.phi += [[(k_v_mat[k][t]+self.beta)/(sum(k_v_mat[k,:])+self.beta*V) for t in range(V)] \
                    for k in range(self.n_topics)]
            self.theta += [ [ (m_k_mat[m][k]+self.alfa)/(sum(m_k_mat[m,:])) for k in range(self.n_topics)]\
                    for m in range(D)]
        self.phi /= stable_times
        self.theta /= stable_times
        return self


if __name__ == "__main__":
    x = np.array([ [random.randint(0,499) for i in range(100)] for j in range(1000)])
    mo = lda(n_topics = 10)
    mo.fit(x)
    mo.phi
    print mo.theta
    print mo.phi
