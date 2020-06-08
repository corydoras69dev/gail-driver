

from rllab.spaces.base import Space
import tensorflow as tf
import numpy as np
from rllab import config
import ipdb


class Product(Space):
    def __init__(self, *components):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        if isinstance(components[0], (list, tuple)):
            assert len(components) == 1
            components = components[0]
        self._components = tuple(components)
        dtypes = [c.new_tensor_variable(
            "tmp", extra_dims=0).dtype for c in components]
        if len(dtypes) > 0 and hasattr(dtypes[0], "as_numpy_dtype"):
            dtypes = [d.as_numpy_dtype for d in dtypes]
        self._common_dtype = np.core.numerictypes.find_common_type([], dtypes)

    def sample(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return tuple(x.sample() for x in self._components)

    @property
    def components(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return self._components

    def contains(self, x):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return isinstance(x, tuple) and all(c.contains(xi) for c, xi in zip(self._components, x))

    def new_tensor_variable(self, name, extra_dims):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return tf.placeholder(
            dtype=self._common_dtype,
            shape=[None] * extra_dims + [self.flat_dim],
            name=name,
        )

    @property
    def flat_dim(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return np.sum([c.flat_dim for c in self._components])

    def flatten(self, x):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return np.concatenate([c.flatten(xi) for c, xi in zip(self._components, x)])

    def flatten_n(self, xs):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        xs_regrouped = [[x[i] for x in xs] for i in range(len(xs[0]))]
        flat_regrouped = [c.flatten_n(xi) for c, xi in zip(
            self.components, xs_regrouped)]
        return np.concatenate(flat_regrouped, axis=-1)

    def unflatten(self, x):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        dims = [c.flat_dim for c in self._components]
        flat_xs = np.split(x, np.cumsum(dims)[:-1])
        return tuple(c.unflatten(xi) for c, xi in zip(self._components, flat_xs))

    def unflatten_n(self, xs):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        dims = [c.flat_dim for c in self._components]
        flat_xs = np.split(xs, np.cumsum(dims)[:-1], axis=-1)
        unflat_xs = [c.unflatten_n(xi)
                     for c, xi in zip(self.components, flat_xs)]
        unflat_xs_grouped = list(zip(*unflat_xs))
        return unflat_xs_grouped

    def __eq__(self, other):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        if not isinstance(other, Product):
            return False
        return tuple(self.components) == tuple(other.components)

    def __hash__(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return hash(tuple(self.components))
