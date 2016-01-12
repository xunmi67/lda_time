import numpy as np


class GibbsSampler:

    """
    gibbs_sampler generator
    """
    def __init__(self, Q):
        """
        sampler generate sample from transition distribution Q,
        Q[0] represent transfer probability to state 0
        :param Q: transition distribution matrix,numpy array
        :return: None
        """
        self.Q = Q

    def sample(self):
        """
        gen a sample
        :return: float number
        """
        pass
