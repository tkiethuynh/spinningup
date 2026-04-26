import numpy as np
import tensorflow as tf
import gymnasium as gym
import time
from copy import deepcopy
from typing import Callable, Optional, Dict, Any
import spinup.algos.tf2.sac.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.device_utils import get_tf_device

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: tf.convert_to_tensor(v) for k,v in batch.items()}

def sac(env_fn: Callable[[], gym.Env], 
        actor_critic: type[tf.keras.Model] = core.MLPActorCritic, 
        ac_kwargs: Dict[str, Any] = dict(), 
        seed: int = 0, 
        steps_per_epoch: int = 4000, 
        epochs: int = 100, 
        replay_size: int = 1000000, 
        gamma: float = 0.99, 
        polyak: float = 0.995, 
        lr: float = 1e-3, 
        alpha: float = 0.2, 
        batch_size: int = 100, 
        start_steps: int = 10000, 
        update_after: int = 1000, 
        update_every: int = 50, 
        num_test_episodes: int = 10, 
        max_ep_len: int = 1000, 
        logger_kwargs: Dict[str, Any] = dict(), 
        save_freq: int = 1):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.random.set_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Device selection
    device = get_tf_device()
    logger.log(f'Using device: {device}')

    with tf.device(device):
        ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        ac_targ = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

        # Initialize variables by calling models
        dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
        dummy_act = np.zeros((1, act_dim), dtype=np.float32)
        ac(dummy_obs, dummy_act)
        ac_targ(dummy_obs, dummy_act)

    def get_vars(model):
        return model.pi.variables + model.q1.variables + model.q2.variables

    ac_vars = get_vars(ac)
    ac_targ_vars = get_vars(ac_targ)

    for v, vt in zip(ac_vars, ac_targ_vars):
        vt.assign(v)

    sync_params(ac_vars)
    sync_params(ac_targ_vars)

    # Setup model saving
    logger.setup_tf2_saver(ac)

    pi_optimizer = MpiAdamOptimizer(learning_rate=lr)
    q_optimizer = MpiAdamOptimizer(learning_rate=lr)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    @tf.function
    def compute_loss_q(data):
        obs, act, rew, obs2, done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        
        q1 = ac.q1(obs, act)
        q2 = ac.q2(obs, act)

        # Bellman backup for Q functions
        a2, logp_a2 = ac.pi(obs2)

        # Target Q-values
        q1_targ = ac_targ.q1(obs2, a2)
        q2_targ = ac_targ.q2(obs2, a2)
        q_targ = tf.minimum(q1_targ, q2_targ)
        backup = tf.stop_gradient(rew + gamma * (1 - done) * (q_targ - alpha * logp_a2))

        # MSE loss against Bellman backup
        loss_q1 = tf.reduce_mean((q1 - backup)**2)
        loss_q2 = tf.reduce_mean((q2 - backup)**2)
        loss_q = loss_q1 + loss_q2
        return loss_q

    @tf.function
    def compute_loss_pi(data):
        obs = data['obs']
        pi, logp_pi = ac.pi(obs)
        q1_pi = ac.q1(obs, pi)
        q2_pi = ac.q2(obs, pi)
        q_pi = tf.minimum(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = tf.reduce_mean(alpha * logp_pi - q_pi)
        return loss_pi

    def update(data):
        with tf.device(device):
            # Q-update
            with tf.GradientTape() as tape:
                loss_q = compute_loss_q(data)
            
            q_trainable_vars = list(ac.q1.trainable_variables) + list(ac.q2.trainable_variables)
            grads = tape.gradient(loss_q, q_trainable_vars)
            q_optimizer.apply_gradients(zip(grads, q_trainable_vars))

            logger.store(LossQ=loss_q)

            # Policy update
            with tf.GradientTape() as tape:
                loss_pi = compute_loss_pi(data)
            grads = tape.gradient(loss_pi, ac.pi.trainable_variables)
            pi_optimizer.apply_gradients(zip(grads, ac.pi.trainable_variables))

            logger.store(LossPi=loss_pi)

            # Polyak averaging
            for v, vt in zip(ac_vars, ac_targ_vars):
                vt.assign(polyak * vt + (1 - polyak) * v)

    def get_action(o, deterministic=False):
        return ac.act(o.reshape(1,-1).astype('float32'), deterministic)[0]

    def test_agent():
        for j in range(num_test_episodes):
            o, _ = test_env.reset()
            d, ep_ret, ep_len = False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                o, r, terminated, truncated, _ = test_env.step(get_action(o, True))
                d = terminated or truncated
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, _ = env.reset()
    ep_ret, ep_len = 0, 0

    total_steps = steps_per_epoch * epochs
    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        o2, r, terminated, truncated, _ = env.step(a)
        d = terminated or truncated
        ep_ret += r
        ep_len += 1

        d = False if ep_len==max_ep_len else d
        replay_buffer.store(o, a, r, o2, d)
        o = o2

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _ = env.reset()
            ep_ret, ep_len = 0, 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            test_agent()

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t+1)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v4')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
