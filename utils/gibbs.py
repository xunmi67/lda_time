import random
import math
import numpy


def sampler(Q):
    """
    gen a sample
    :return: float number
    """
    Q = numpy.array(Q)
    acc_Q = Q.copy()
    for i in xrange(1,len(acc_Q)):
        acc_Q[i] += acc_Q[i-1]
    assert math.fabs(acc_Q[-1] - 1.0) < 0.01
    rd = random.random()
    for i in xrange(len(acc_Q)):
        if acc_Q[i] > rd:
            return i
    # this code should never execute
    assert 1 > 2
    pass

if __name__ == "__main__":
    Q = [0.1,0.1,0.6,0.1,0.1]
    print([sampler(Q) for i in xrange(100)])
