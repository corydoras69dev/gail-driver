from rllab.sampler.base import BaseSampler
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
import tensorflow as tf
from rllab import config
import ipdb


def worker_init_tf(G):
    if config.TF_NN_SETTRACE:
        ipdb.set_trace()
    G.sess = tf.Session()
    G.sess.__enter__()


def worker_init_tf_vars(G):
    if config.TF_NN_SETTRACE:
        ipdb.set_trace()
    G.sess.run(tf.initialize_all_variables())


class BatchSampler(BaseSampler):
    def start_worker(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        parallel_sampler.populate_task(self.algo.env, self.algo.policy)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        if config.TF_NN_SETTRACE:
            ipdb.set_trace()
        cur_policy_params = self.algo.policy.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_policy_params,
            env_params=None,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(
                paths, self.algo.batch_size)
            return paths_truncated
