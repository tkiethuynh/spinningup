import time
from typing import Callable, Dict, Any

import gymnasium as gym
import numpy as np

from spinup.utils.device_utils import setup_tf_gpu, get_tf_device
setup_tf_gpu()

import tensorflow as tf

import spinup.algos.tf2.td3.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_params


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: tf.convert_to_tensor(v) for k, v in batch.items()}


def td3(env_fn: Callable[[], gym.Env],
        actor_critic: type[tf.keras.Model] = core.MLPActorCritic,
        ac_kwargs: Dict[str, Any] = dict(),
        seed: int = 0,
        steps_per_epoch: int = 4000,
        epochs: int = 100,
        replay_size: int = 1000000,
        gamma: float = 0.99,
        polyak: float = 0.995,
        pi_lr: float = 1e-3,
        q_lr: float = 1e-3,
        batch_size: int = 100,
        start_steps: int = 10000,
        update_after: int = 1000,
        update_every: int = 50,
        act_noise: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        num_test_episodes: int = 10,
        max_ep_len: int = 1000,
        logger_kwargs: Dict[str, Any] = dict(),
        save_freq: int = 1,
        device: str = 'cpu'):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) (TensorFlow 2)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the Gymnasium API.

        actor_critic: The constructor method for a TensorFlow 2 Keras Model
            with an ``act`` method, a ``pi`` submodel, a ``q1`` submodel,
            and a ``q2`` submodel. The ``act`` method and ``pi`` submodel
            should accept batches of observations as inputs, and ``q1`` and
            ``q2`` should accept a batch of observations and a batch of actions
            as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target
            policy.

        noise_clip (float): Limit for absolute value of target policy
            smoothing noise.

        policy_delay (int): Policy will only be updated once every
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        device (str): Compute device (`'cpu'`, `'gpu'`, `'cuda'`, `'/GPU:0'`,
            or `'auto'`). Defaults to `'cpu'`, which is usually fastest for
            single-environment rollouts with shallow MLPs; use `'auto'` or
            `'gpu'` for larger networks or GPU setups.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed (set_random_seed also resets Keras 3 layer initializer RNG)
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError(
            f"TD3 requires a continuous (gymnasium.spaces.Box) action space, "
            f"got {type(env.action_space).__name__} ({env.action_space}). "
            f"For discrete action spaces, use vpg, trpo, or ppo."
        )
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Select compute device
    tf_device = get_tf_device(device)
    logger.log(f'Using device: {tf_device}')

    # Construct main and target actor-critic models and materialize weights via dummy pass
    with tf.device(tf_device):
        ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        ac_targ = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

        dummy_obs = np.zeros((1, *obs_dim), dtype=np.float32)
        dummy_act = np.zeros((1, act_dim), dtype=np.float32)
        ac(dummy_obs, dummy_act)
        ac_targ(dummy_obs, dummy_act)

        # Initialize target parameters equal to main parameters
        for v, vt in zip(ac.variables, ac_targ.variables):
            vt.assign(v)

        # Combine twin Q trainable variables and build optimizers eagerly before @tf.function
        q_trainable_vars = list(ac.q1.trainable_variables) + list(ac.q2.trainable_variables)
        pi_optimizer = MpiAdamOptimizer(learning_rate=pi_lr)
        q_optimizer = MpiAdamOptimizer(learning_rate=q_lr)
        pi_optimizer.build(ac.pi.trainable_variables)
        q_optimizer.build(q_trainable_vars)

    sync_params(ac.variables)
    sync_params(ac_targ.variables)

    # Set up model saving (.keras archive)
    logger.setup_tf2_saver(ac)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Compiled @tf.function Q-update: clipped double-Q + target policy smoothing
    @tf.function
    def train_q_step(obs, act, rew, obs2, done):
        with tf.GradientTape() as tape:
            q1 = ac.q1(obs, act)
            q2 = ac.q2(obs, act)

            # Target policy smoothing
            eps = tf.random.normal(tf.shape(act)) * target_noise
            eps = tf.clip_by_value(eps, -noise_clip, noise_clip)
            a_targ = tf.clip_by_value(ac_targ.pi(obs2) + eps, -act_limit, act_limit)

            # Target Q-values (min over twin target Q-networks)
            q1_targ = ac_targ.q1(obs2, a_targ)
            q2_targ = ac_targ.q2(obs2, a_targ)
            q_targ = tf.minimum(q1_targ, q2_targ)
            backup = tf.stop_gradient(rew + gamma * (1 - done) * q_targ)

            # MSE losses against Bellman backup
            loss_q = tf.reduce_mean((q1 - backup) ** 2) + tf.reduce_mean((q2 - backup) ** 2)

        grads = tape.gradient(loss_q, q_trainable_vars)
        q_optimizer.apply_gradients(zip(grads, q_trainable_vars))
        return loss_q, q1, q2

    # Compiled @tf.function delayed policy update + Polyak target averaging
    @tf.function
    def train_pi_and_target_step(obs):
        with tf.GradientTape() as tape:
            loss_pi = -tf.reduce_mean(ac.q1(obs, ac.pi(obs)))
        grads = tape.gradient(loss_pi, ac.pi.trainable_variables)
        pi_optimizer.apply_gradients(zip(grads, ac.pi.trainable_variables))

        # Polyak averaging update
        for v, vt in zip(ac.variables, ac_targ.variables):
            vt.assign(polyak * vt + (1.0 - polyak) * v)
        return loss_pi

    def update(data, timer):
        with tf.device(tf_device):
            # First run one gradient descent step for Q1 and Q2
            loss_q, q1, q2 = train_q_step(
                data['obs'], data['act'], data['rew'], data['obs2'], data['done']
            )
            logger.store(LossQ=float(loss_q), Q1Vals=q1.numpy(), Q2Vals=q2.numpy())

            # Possibly update pi and target networks
            if timer % policy_delay == 0:
                loss_pi = train_pi_and_target_step(data['obs'])
                logger.store(LossPi=float(loss_pi))

    def get_action(o, noise_scale):
        with tf.device(tf_device):
            a = ac.act(o.reshape(1, -1).astype('float32'))[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for _ in range(num_test_episodes):
            o, _ = test_env.reset()
            d, ep_ret, ep_len = False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, terminated, truncated, _ = test_env.step(get_action(o, 0))
                d = terminated or truncated
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    start_time = time.time()
    o, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    test_env.reset(seed=seed + 10000)
    test_env.action_space.seed(seed + 10000)
    ep_ret, ep_len = 0, 0

    # Main loop: collect experience in env and update/log each epoch
    total_steps = steps_per_epoch * epochs
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, terminated, truncated, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Store true termination only; never zero bootstrap on truncation or time limit
        replay_buffer.store(o, a, r, o2, terminated)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if terminated or truncated or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _ = env.reset()
            ep_ret, ep_len = 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t + 1)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    from spinup.utils.run_utils import _make_env, setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    td3(lambda: _make_env(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, device=args.device)
