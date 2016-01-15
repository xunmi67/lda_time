import numpy as np
import scipy.sparse as sp
import random
from ..utils import gibbs
class lda():
    MAX_LEN_OF_DOC = 1000
    def __init__(self,n_topics=10,alfa=None,beta=None):
        self.n_topics = n_topics
        self.alfa = 50.0/n_topics if alfa is None else alfa
        self.beta = 0.001 if beta is None else beta

    def _cal_q(self,k_v_mat,m_k_mat,d,w,z,i):
        #q = []
        #for k in range(self.n_topics):
        #    q.append((m_k_mat[d,k]+self.alfa)*(k_v_mat[k,w[i]]+self.beta)/
        #            (k_v_mat[k].sum()+self.beta*self.V))
        q = [ (m_k_mat[d,k]+self.alfa)*(k_v_mat[k,w[i]]+self.beta)/
                    self._sum_k[k] for k in xrange(self.n_topics)]
        #q = np.array(q)
        q = q / sum(q)
        return q

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
        self.D  = X.shape[0]
        self.V = X.max()+1
        D = self.D
        V = self.V
        z_len = [X.indptr[i+1]-X.indptr[i] for i in range(D)]
        z_vectors = [[random.randint(1,self.n_topics-1) for i in range(z_len[m])] for m in range(D)]
        k_v_mat = np.zeros((self.n_topics,V),np.int32)
        m_k_mat = np.zeros((D,self.n_topics),np.int32)
        print("staring initial Z")
        for d in range(D):
            w_d = X.data[X.indptr[d]:X.indptr[d+1]]
            z_d = z_vectors[d]
            assert len(w_d) == len(z_d)
            for i in range(len(w_d)):
                w_i = w_d[i]
                z_i = z_d[i]
                k_v_mat[z_i,w_i] += 1
                m_k_mat[d,z_i] += 1
        self._sum_k = [self.V*self.beta+k_v_mat[k].sum() for k in xrange(self.n_topics)]

        # gibbs sample,burn in process
        print("start buring...")
        import time
        b_time = 0
        for sample_time in range(self.n_topics*self.MAX_LEN_OF_DOC*10):
            for d in range(1,D):
                w_d = X.data[X.indptr[d]:X.indptr[d+1]]
                z_d = z_vectors[d]
                for i in range(len(w_d)):
                    k_v_mat[z_d[i],w_d[i]] -= 1
                    m_k_mat[d,z_d[i]] -= 1
                    self._sum_k[z_d[i]] -= 1
                    # cal sample probability Q
                    if sample_time ==0 and d == 1:
                        print("benchmach:")
                        t0 = time.time()
                        for bi in range(10000):
                            self._cal_q(k_v_mat,m_k_mat,d,w_d,z_d,i)
                        time_of_cal_q = time.time()-t0
                        print("time of cal q:",time_of_cal_q)
                        Q = self._cal_q(k_v_mat,m_k_mat,d,w_d,z_d,i)
                        print(sum(Q))
                        t1 = time.time()
                        for bi in range(10000):
                            gibbs.sampler(Q)
                        time_of_sample = time.time() - t1
                        print("time of sample:",time_of_sample)

                    Q = self._cal_q(k_v_mat,m_k_mat,d,w_d,z_d,i)
                    z_i = gibbs.sampler(Q)
                    k_v_mat[z_i,w_d[i]] += 1
                    m_k_mat[d,z_i] += 1
                    self._sum_k[z_i] += 1
            if b_time % 10 == 0:
                print(b_time+1," of ",self.n_topics*self.MAX_LEN_OF_DOC*20," is over")
            b_time += 1

        # it should come to stable state,state sample, and calculate theta and phi
        self.phi = np.zeros((self.n_topics,V))
        self.theta = np.zeros((D,self.n_topics))
        stable_times = self.n_topics*self.MAX_LEN_OF_DOC*20
        for sample_time in range(stable_times):
            for d in D:
                w_d = X.indices[X.indptr[d],X.indptr[d+1]]
                z_d = z_vectors[d]
                for i in range(w_d):
                    k_v_mat[z_d[i],w_d[i]] -= 1
                    m_k_mat[d,z_d[i]] -= 1
                    # cal sample probability Q
                    Q = []
                    z_i = i
                    k_v_mat[z_i,w_d[i]] += 1
                    m_k_mat[d,z_i] += 1
            self.phi += [[(k_v_mat[k][t]+self.beta)/(sum(k_v_mat[k,:])+self.beta*V) for t in range(V)] \
                    for k in range(self.n_topics)]
            self.theta += [ [ (m_k_mat[m][k]+self.alfa)/(sum(m_k_mat[m,:])) for k in range(self.n_topics)]\
                    for m in range(D)]
        self.phi /= stable_times
        self.theta /= stable_times
        return self


def test():
    x = np.array([ [random.randint(0,499) for i in range(100)] for j in range(1000)])
    mo = lda(n_topics = 10)
    mo.fit(x)
    mo.phi
    print mo.theta
    print mo.phi
