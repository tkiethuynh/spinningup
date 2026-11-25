import time
from typing import Callable, Dict, Any

import gymnasium as gym
import numpy as np

from spinup.utils.device_utils import setup_tf_gpu, get_tf_device
setup_tf_gpu()

import tensorflow as tf

import spinup.algos.tf2.trpo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class TRPOBuffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]


def trpo(env_fn: Callable[[], gym.Env],
         actor_critic: type[tf.keras.Model] = core.MLPActorCritic,
         ac_kwargs: Dict[str, Any] = dict(),
         seed: int = 0,
         steps_per_epoch: int = 4000,
         epochs: int = 50,
         gamma: float = 0.99,
         delta: float = 0.01,
         vf_lr: float = 1e-3,
         train_v_iters: int = 80,
         damping_coeff: float = 0.1,
         cg_iters: int = 10,
         backtrack_iters: int = 10,
         backtrack_coeff: float = 0.8,
         lam: float = 0.97,
         max_ep_len: int = 1000,
         logger_kwargs: Dict[str, Any] = dict(),
         save_freq: int = 10,
         device: str = 'cpu'):
    """
    Trust Region Policy Optimization (TensorFlow 2)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the Gymnasium API.

        actor_critic: The constructor method for a TensorFlow 2 Keras Model
            with a ``step`` method, an ``act`` method, a ``pi`` submodel,
            and a ``v`` submodel. The ``step`` method should accept a batch
            of observations and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns
            ``a`` (optionally supporting ``deterministic=True``).

            The ``pi`` submodel outputs distribution parameters (logits for
            discrete actions, or a ``(mu, std)`` tuple for continuous actions)
            and provides ``log_prob_from_distribution``, ``entropy``, and
            analytic ``kl_divergence``.

            The ``v`` submodel's call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to TRPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        delta (float): KL-divergence limit for TRPO update.
            (Should be small for stability. Values like 0.01, 0.05.)

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        damping_coeff (float): Artifact for numerical stability, should be
            smallish. Adjusts Hessian-vector product calculation:

            .. math:: Hv \\rightarrow (\\alpha I + H)v

            where :math:`\\alpha` is the damping coefficient.
            Probably don't play with this hyperparameter.

        cg_iters (int): Number of iterations of conjugate gradient to perform.
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down.

            Also probably don't play with this hyperparameter.

        backtrack_iters (int): Maximum number of steps allowed in the
            backtracking line search. Since the line search usually doesn't
            backtrack, and usually only steps back once when it does, this
            hyperparameter doesn't often matter.

        backtrack_coeff (float): How far back to step during backtracking line
            search. (Always between 0 and 1, usually above 0.5.)

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        device (str): Compute device (`'cpu'`, `'gpu'`, `'cuda'`, `'/GPU:0'`,
            or `'auto'`). Defaults to `'cpu'`, which is usually fastest for
            single-environment rollouts with shallow MLPs; use `'auto'` or
            `'gpu'` for larger networks or GPU setups.

    """

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed (set_random_seed also resets Keras 3 layer initializer RNG)
    seed += 10000 * proc_id()
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Select compute device
    tf_device = get_tf_device(device)
    logger.log(f'Using device: {tf_device}')

    # Construct actor-critic Keras model and materialize weights with a dummy pass
    with tf.device(tf_device):
        ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        dummy_obs = np.zeros((1, *obs_dim), dtype=np.float32)
        ac(dummy_obs)

    # Sync params across MPI processes so all workers start identically
    sync_params(ac.variables)

    # Set up model saving (.keras archive)
    logger.setup_tf2_saver(ac)

    # Build value optimizer eagerly before @tf.function tracing
    with tf.device(tf_device):
        vf_optimizer = MpiAdamOptimizer(learning_rate=vf_lr)
        vf_optimizer.build(ac.v.trainable_variables)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = TRPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    def flat_concat(xs):
        """Flatten and concatenate a sequence of tensors into a 1D vector."""
        return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)

    def assign_params_from_flat(x, params):
        """Unflatten a 1D parameter vector back into trainable variables."""
        flat_size = lambda p: int(np.prod(p.shape))
        splits = tf.split(x, [flat_size(p) for p in params])
        for p, p_new in zip(params, splits):
            p.assign(tf.reshape(p_new, p.shape))

    def get_pi_loss(obs, act, adv, logp_old):
        """Compute TRPO surrogate policy loss and entropy."""
        if isinstance(ac.pi, core.MLPCategoricalActor):
            logits = ac.pi(obs)
            logp = ac.pi.log_prob_from_distribution(logits, tf.cast(act, tf.int32))
            ent = ac.pi.entropy(logits)
        else:
            mu_std = ac.pi(obs)
            logp = ac.pi.log_prob_from_distribution(mu_std, act)
            ent = ac.pi.entropy(mu_std)
        return -tf.reduce_mean(tf.exp(logp - logp_old) * adv), logp, tf.reduce_mean(ent)

    def get_detached_dist(obs):
        """Snapshot old policy distribution pi_{theta_k} held fixed during HVP and line search."""
        if isinstance(ac.pi, core.MLPCategoricalActor):
            return tf.stop_gradient(ac.pi(obs))
        mu, std = ac.pi(obs)
        return (tf.stop_gradient(mu), tf.stop_gradient(std))

    def kl_from_old(obs, old_dist):
        """Mean forward KL divergence D_KL(pi_{theta_k} || pi_theta) over batch."""
        if isinstance(ac.pi, core.MLPCategoricalActor):
            logits = ac.pi(obs)
            return tf.reduce_mean(ac.pi.kl_divergence(old_dist, logits))
        mu_std = ac.pi(obs)
        return tf.reduce_mean(ac.pi.kl_divergence(old_dist, mu_std))

    # Precompute offsets into flat vector for slicing during Hessian-vector product
    var_indices = []
    curr = 0
    for v in ac.pi.trainable_variables:
        size = int(np.prod(v.shape))
        var_indices.append(curr)
        curr += size

    # Nested GradientTape computes exact Fisher-vector product H v = grad(grad(KL) . v) + alpha v
    @tf.function
    def compute_hvp(obs, v, old_dist):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                kl = kl_from_old(obs, old_dist)
            grads = t1.gradient(kl, ac.pi.trainable_variables)
            grad_v = tf.add_n([
                tf.reduce_sum(
                    g * tf.reshape(v[idx_start:idx_start + int(np.prod(g.shape))], g.shape)
                )
                for idx_start, g in zip(var_indices, grads)
            ])

        hvp_list = t2.gradient(grad_v, ac.pi.trainable_variables)
        return flat_concat(hvp_list) + damping_coeff * v

    @tf.function
    def compute_loss_v(obs, ret):
        return tf.reduce_mean((ret - ac.v(obs)) ** 2)

    def cg(hvp_fn, b):
        """
        Conjugate gradient algorithm (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        for approximately solving Hx = b without materializing the full Fisher matrix H.
        """
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        rdotr = np.dot(r, r)
        for _ in range(cg_iters):
            z = hvp_fn(p)
            alpha = rdotr / (np.dot(p, z) + 1e-8)
            x += alpha * p
            r -= alpha * z
            new_rdotr = np.dot(r, r)
            if np.sqrt(new_rdotr) < 1e-10:
                break
            mu = new_rdotr / (rdotr + 1e-8)
            p = r + mu * p
            rdotr = new_rdotr
        return x

    def update():
        obs, act, adv, ret, logp_old = [tf.convert_to_tensor(x) for x in buf.get()]

        with tf.device(tf_device):
            old_dist = get_detached_dist(obs)
            v_l_old = compute_loss_v(obs, ret)

            # Policy gradient g wrt current parameters (averaged across MPI ranks)
            with tf.GradientTape() as tape:
                loss_pi, logp, ent = get_pi_loss(obs, act, adv, logp_old)

            loss_pi_old = float(mpi_avg(loss_pi.numpy()))
            ent_val = float(mpi_avg(ent.numpy()))

            grads = tape.gradient(loss_pi, ac.pi.trainable_variables)
            g = flat_concat(grads).numpy()
            g = mpi_avg(g)

            def hvp(v):
                hvp_res = compute_hvp(obs, tf.convert_to_tensor(v, dtype=tf.float32), old_dist)
                return mpi_avg(hvp_res.numpy())

            # Solve H x ~= g via conjugate gradients and compute max step size for KL <= delta
            x = cg(hvp, g)
            alpha_val = np.sqrt(2 * delta / (np.dot(x, hvp(x)) + 1e-8))
            old_params = flat_concat(ac.pi.trainable_variables).numpy()

            def check_constraints(step):
                assign_params_from_flat(
                    tf.convert_to_tensor(old_params - step, dtype=tf.float32),
                    ac.pi.trainable_variables,
                )
                kl_cand = float(mpi_avg(kl_from_old(obs, old_dist).numpy()))
                loss_cand = float(mpi_avg(get_pi_loss(obs, act, adv, logp_old)[0].numpy()))
                return kl_cand <= 1.5 * delta and loss_cand <= loss_pi_old

            # Backtracking line search to guarantee trust-region KL bound and improvement
            accepted_iter = backtrack_iters
            for i in range(backtrack_iters):
                step = alpha_val * x * (backtrack_coeff ** i)
                if check_constraints(step):
                    accepted_iter = i
                    break
            else:
                assign_params_from_flat(
                    tf.convert_to_tensor(old_params, dtype=tf.float32),
                    ac.pi.trainable_variables,
                )

            # Value function learning
            for _ in range(train_v_iters):
                with tf.GradientTape() as tape:
                    loss_v = compute_loss_v(obs, ret)
                grads = tape.gradient(loss_v, ac.v.trainable_variables)
                vf_optimizer.apply_gradients(zip(grads, ac.v.trainable_variables))

            # Log changes from update
            loss_pi_new, _, _ = get_pi_loss(obs, act, adv, logp_old)
            loss_v_new = compute_loss_v(obs, ret)
            kl_final = kl_from_old(obs, old_dist)
            logger.store(
                LossPi=loss_pi_old,
                LossV=float(v_l_old),
                KL=float(kl_final),
                Entropy=ent_val,
                DeltaLossPi=float(loss_pi_new) - loss_pi_old,
                DeltaLossV=float(loss_v_new - v_l_old),
                BacktrackIters=accepted_iter,
            )

    # Prepare for interaction with environment
    start_time = time.time()
    o, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    ep_ret, ep_len = 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            with tf.device(tf_device):
                a, v, logp = ac.step(o.reshape(1, -1).astype('float32'))

            next_o, r, terminated, truncated, _ = env.step(a[0])
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a[0], r, v[0], logp[0])
            logger.store(VVals=v[0])

            # Update obs (critical!)
            o = next_o

            timeout = (ep_len == max_ep_len) or truncated
            terminal = terminated or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                # Only zero bootstrap on true termination; bootstrap on truncation/horizon/epoch cutoff
                with tf.device(tf_device):
                    last_val = (
                        0.0
                        if terminated
                        else float(ac.v(o.reshape(1, -1).astype('float32')).numpy()[0])
                    )
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, _ = env.reset()
                ep_ret, ep_len = 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Perform TRPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('BacktrackIters', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='trpo')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import _make_env, setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    trpo(lambda: _make_env(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs, device=args.device)
