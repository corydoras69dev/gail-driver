import time
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from tf_rllab.policies.base import Policy
import tensorflow as tf
from tf_rllab.samplers.batch_sampler import BatchSampler
import seedmng.mng
from rltools.envs.julia_sim import JuliaEnvWrapper, JuliaEnv
import random
import numpy as np
import pdb
import julia
import joblib
from rllab import config

class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            load_params_args=None,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.load_params_args = load_params_args
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        if sampler_cls is None:
            sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.temporal_noise_thresh = kwargs['temporal_noise_thresh']
        self.init_opt()

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr):
        return self.sampler.obtain_samples(itr)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths)

    def train(self):
        sm = seedmng.mng.SeedMng()
        j = julia.Julia()
        j.using("Base.Random.srand")
        time.sleep(8)
        #pdb.set_trace()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            if self.load_params_args is not None:
                self.policy.load_params(*self.load_params_args)
            self.start_worker()
            start_time = time.time()

            if self.start_itr is not 0:
                #pdb.set_trace()
                tf_filename = config.LOG_DIR + "/tf_" + str(self.start_itr) +".ckpt"
                saver.restore(sess, tf_filename)

            for itr in range(self.start_itr, self.n_itr):

                sm.set_iteration(itr)
                random.seed(sm.get_system_seed(0))
                np.random.seed(seed=sm.get_np_seed(0))
                tf.set_random_seed(sm.get_tf_system_seed(0))
                j.srand(sm.get_system_seed(0))
                time.sleep(2)

                #pdb.set_trace()
                self.policy.save_params(itr)
                itr_start_time = time.time()
                if itr >= self.temporal_noise_thresh:
                    self.env._noise_indicies = None

                with logger.prefix('itr #%d | ' % itr):
                    logger.log("Obtaining samples...")
                    #pdb.set_trace()
                    paths = self.obtain_samples(itr)
                    logger.log("Processing samples...")
                    samples_data = self.process_samples(itr, paths)
                    logger.log("Logging diagnostics...")
                    # env, policy, baseline have individual log_diagnos methods
                    # for overriding
                    self.log_diagnostics(paths)
                    logger.log("Optimizing policy...")
                    self.optimize_policy(itr, samples_data)
                    logger.log("Saving snapshot...")
                    # pdb.set_trace()
                    params = self.get_itr_snapshot(
                        itr, samples_data)  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = samples_data["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular(
                        'ItrTime', time.time() - itr_start_time)
                    logger.dump_tabular(with_prefix=False)
                    saver.save(sess, logger.get_snapshot_dir() + "/tf_" + str(itr + 1) + ".ckpt")
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                  "continue...")
        self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
