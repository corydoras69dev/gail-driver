from contextlib import contextmanager

from rllab.core.serializable import Serializable
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
import tensorflow as tf
import numpy as np
import ipdb
import h5py
import os
from rllab import config
import ipdb

load_params = True


@contextmanager
def suppress_params_loading():
    if config.TF_NN_SETTRACE:
        ipdb.set_trace()
    global load_params
    load_params = False
    yield
    load_params = True


class Parameterized(object):
    def __init__(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        self._cached_params = {}
        self._cached_param_dtypes = {}
        self._cached_param_shapes = {}
        self._cached_assign_ops = {}
        self._cached_assign_placeholders = {}
        self.save_name = 'policy_gail'

    def get_params_internal(self, **tags):
        """
        Internal method to be implemented which does not perform caching
        """
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        raise NotImplementedError

    def get_params(self, **tags):
        """
        Get the list of parameters, filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(**tags)
        return self._cached_params[tag_tuple]

    def get_param_dtypes(self, **tags):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [
                val.dtype for val in param_values]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, **tags):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [
                val.shape for val in param_values]
        return self._cached_param_shapes[tag_tuple]

    def get_param_values(self, **tags):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        params = self.get_params(**tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def set_param_values(self, flattened_params, **tags):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(**tags))
        ops = []
        feed_dict = dict()
        for param, dtype, value in zip(
                self.get_params(**tags),
                self.get_param_dtypes(**tags),
                param_values):
            if param not in self._cached_assign_ops:
                assign_placeholder = tf.placeholder(
                    dtype=param.dtype.base_dtype)
                assign_op = tf.assign(param, assign_placeholder)
                self._cached_assign_ops[param] = assign_op
                self._cached_assign_placeholders[param] = assign_placeholder
            ops.append(self._cached_assign_ops[param])
            feed_dict[self._cached_assign_placeholders[param]
                      ] = value.astype(dtype)
            if debug:
                print("setting value of %s" % param.name)
        tf.get_default_session().run(ops, feed_dict=feed_dict)

    def flat_to_params(self, flattened_params, **tags):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return unflatten_tensors(flattened_params, self.get_param_shapes(**tags))

    def __getstate__(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        d = Serializable.__getstate__(self)
        global load_params
        if load_params:
            d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            tf.get_default_session().run(tf.initialize_variables(self.get_params()))
            self.set_param_values(d["params"])


class JointParameterized(Parameterized):
    def __init__(self, components):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        super(JointParameterized, self).__init__()
        self.components = components

    def get_params_internal(self, **tags):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        params = [
            param for comp in self.components for param in comp.get_params_internal(**tags)]
        # only return unique parameters
        return sorted(set(params), key=hash)


class Model(Parameterized):
    _load_dir = './models'
    _log_dir = './models'

    def load_params(self, filename, itr, skip_params):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        #ipdb.set_trace()
        print 'loading policy params...'
        if not hasattr(self, 'load_dir'):
            log_dir = Model._load_dir
        else:
            log_dir = self.load_dir
        filename = log_dir + "/" + filename + '.h5'
        assignments = []

        with h5py.File(filename, 'r') as hf:
            if itr >= 0:
                prefix = self._prefix(itr)
            else:
                prefix = hf.keys()[itr] + "/"

            for param in self.get_params():
                path = prefix + param.name
                if param.name in skip_params:
                    continue

                if path in hf:
                    assignments.append(
                        param.assign(hf[path][...])
                    )

        sess = tf.get_default_session()
        sess.run(assignments)
        print 'done.'

    def restore_params(self, filename):
        print 'loading policy params...'
#        assignments = []
#        with h5py.File(filename, 'r') as hf:
#            for param in self.get_params():
#                if path in hf:
#                    assignments.append(
#                        param.assign(hf[path][...])
#                    )
#
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        sess = tf.get_default_session()
        saver = tf.train.Saver()
        saver.restore(sess, filename)
#        sess.run(assignments)
        print 'done.'

    def save_params(self, itr, type_gru, overwrite=False):
        print 'saving model...'
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        if not hasattr(self, 'log_dir'):
            log_dir = Model._log_dir
        else:
            log_dir = self.log_dir
        filename = log_dir + "/" + self.save_name + '.h5'
        sess = tf.get_default_session()

        key = self._prefix(itr)
        with h5py.File(filename, 'a') as hf:
            if key in hf:
                dset = hf[key]
            else:
                dset = hf.create_group(key)

            vs = self.get_params()
            vals = sess.run(vs)

            for v, val in zip(vs, vals):
                dset[v.name] = val

        if type_gru:
            type_name = '_gru'
        else:
            type_name = '_mlp'
        filename = log_dir + "/" + self.save_name + type_name + '-' + str(itr) + '.h5'
        sess = tf.get_default_session()

        key = self._prefix(itr)
        with h5py.File(filename, 'w') as hf:
            if key in hf:
                dset = hf[key]
            else:
                dset = hf.create_group(key)

            vs = self.get_params()
            vals = sess.run(vs)

            for v, val in zip(vs, vals):
                dset[v.name] = val

        print 'done.'
        pass

    def write_params(self, filename):
        print 'saving model...'
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        sess = tf.get_default_session()
        saver = tf.train.Saver(max_to_keep=None)
        saver.save(sess, filename)
#        with h5py.File(filename, "w") as hf:
#            vs = self.get_params()
#            vals = sess.run(vs)
#            for v, val in zip(vs, vals):
#                hf.create_dataset(v.name, data=val)
        print 'done.'

    def save_extra_data(self, names, data):
        print 'saving model...'
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        if not hasattr(self, 'log_dir'):
            log_dir = Model._log_dir
        else:
            log_dir = self.log_dir
        filename = log_dir + "/" + self.save_name + '.h5'
        sess = tf.get_default_session()

        assert len(names) == len(data)
        with h5py.File(filename, 'a') as hf:
            for name, d in zip(names, data):
                hf.create_dataset(name, data=d)

        print 'done.'

    def set_log_dir(self, log_dir):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        self.log_dir = log_dir

    def set_load_dir(self, load_dir):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        self.load_dir = load_dir

    @staticmethod
    def _prefix(x):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        return 'iter{:05}/'.format(x)
