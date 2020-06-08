

from rllab.spaces.box import Box as TheanoBox
import tensorflow as tf
from rllab import config
import ipdb


class Box(TheanoBox):
    def new_tensor_variable(self, name, extra_dims):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.flat_dim], name=name)
