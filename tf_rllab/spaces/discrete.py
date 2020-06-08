from rllab.spaces.base import Space
import numpy as np
from rllab.misc import special
from rllab.misc import ext
import tensorflow as tf
from rllab import config
import ipdb


class Discrete(Space):
    """
    {0,1,...,n-1}
    """

    def __init__(self, n):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        self._n = n

    @property
    def n(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return self._n

    def sample(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return np.random.randint(self.n)

    def sample_n(self, n):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return np.random.randint(low=0, high=self.n, size=n)

    def contains(self, x):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        x = np.asarray(x)
        return x.shape == () and x.dtype.kind == 'i' and x >= 0 and x < self.n

    def __repr__(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return self.n == other.n

    def flatten(self, x):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return special.to_onehot(x, self.n)

    def unflatten(self, x):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return special.from_onehot(x)

    def flatten_n(self, x):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return special.to_onehot_n(x, self.n)

    def unflatten_n(self, x):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return special.from_onehot_n(x)

    @property
    def default_value(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return 0

    @property
    def flat_dim(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return self.n

    def weighted_sample(self, weights):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return special.weighted_sample(weights, range(self.n))

    def new_tensor_variable(self, name, extra_dims):
        # needed for safe conversion to float32
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return tf.placeholder(dtype=tf.uint8, shape=[None] * extra_dims + [self.flat_dim], name=name)

    def __eq__(self, other):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        if not isinstance(other, Discrete):
            return False
        return self.n == other.n

    def __hash__(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return hash(self.n)
