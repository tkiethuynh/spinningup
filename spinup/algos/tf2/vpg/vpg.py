import numpy as np
import tensorflow as tf
import gymnasium as gym
import time
from typing import Callable, Optional, Dict, Any
import spinup.algos.tf2.vpg.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.device_utils import get_tf_device

class VPGBuffer:
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
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.combined_shape(0) # placeholder logic
        # Re-using the core.discount_cumsum logic from original
        import scipy.signal
        def discount_cumsum(x, discount):
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]

def vpg(env_fn: Callable[[], gym.Env], 
        actor_critic: type[tf.keras.Model] = core.MLPActorCritic, 
        ac_kwargs: Dict[str, Any] = dict(), 
        seed: int = 0, 
        steps_per_epoch: int = 4000, 
        epochs: int = 50, 
        gamma: float = 0.99, 
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3, 
        train_v_iters: int = 80, 
        lam: float = 0.97, 
        max_ep_len: int = 1000,
        logger_kwargs: Dict[str, Any] = dict(), 
        save_freq: int = 10):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.random.set_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    # Device selection
    device = get_tf_device()
    logger.log(f'Using device: {device}')

    with tf.device(device):
        ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

        # Build model
        dummy_obs = np.zeros((1, obs_dim[0]), dtype=np.float32)
        ac(dummy_obs)

    # Sync params across processes
    sync_params(ac.variables)

    # Setup model saving
    logger.setup_tf2_saver(ac)

    # Set up optimizers
    pi_optimizer = MpiAdamOptimizer(learning_rate=pi_lr)
    vf_optimizer = MpiAdamOptimizer(learning_rate=vf_lr)

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    @tf.function
    def compute_loss_pi(obs, act, adv):
        if isinstance(ac.pi, core.MLPCategoricalActor):
            logits = ac.pi(obs)
            logp = ac.pi.log_prob_from_distribution(logits, tf.cast(act, tf.int32))
        else:
            mu, std = ac.pi(obs)
            logp = ac.pi.log_prob_from_distribution((mu, std), act)
        loss_pi = -tf.reduce_mean(logp * adv)
        return loss_pi, logp

    @tf.function
    def compute_loss_v(obs, ret):
        return tf.reduce_mean((ret - ac.v(obs))**2)

    def update():
        obs, act, adv, ret, logp_old = [tf.convert_to_tensor(x) for x in buf.get()]

        with tf.device(device):
            # Policy update
            with tf.GradientTape() as tape:
                loss_pi, logp = compute_loss_pi(obs, act, adv)
            
            grads = tape.gradient(loss_pi, ac.pi.trainable_variables)
            pi_optimizer.apply_gradients(zip(grads, ac.pi.trainable_variables))

            # Value update
            for _ in range(train_v_iters):
                with tf.GradientTape() as tape:
                    loss_v = compute_loss_v(obs, ret)
                grads = tape.gradient(loss_v, ac.v.trainable_variables)
                vf_optimizer.apply_gradients(zip(grads, ac.v.trainable_variables))

            # Log changes
            loss_pi_new, _ = compute_loss_pi(obs, act, adv)
            loss_v_new = compute_loss_v(obs, ret)
            kl = tf.reduce_mean(logp_old - logp)
            logger.store(LossPi=loss_pi, LossV=loss_v, 
                         KL=kl, Entropy=tf.reduce_mean(-logp), 
                         DeltaLossPi=(loss_pi_new - loss_pi),
                         DeltaLossV=(loss_v_new - loss_v))

    start_time = time.time()
    o, _ = env.reset()
    ep_ret, ep_len = 0, 0

    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(o.reshape(1,-1).astype('float32'))

            next_o, r, terminated, truncated, _ = env.step(a[0])
            d = terminated or truncated
            ep_ret += r
            ep_len += 1

            buf.store(o, a[0], r, v[0], logp[0])
            logger.store(VVals=v[0])

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                last_val = 0 if d else ac.v(o.reshape(1,-1).astype('float32')).numpy()[0]
                buf.finish_path(last_val)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, _ = env.reset()
                ep_ret, ep_len = 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Update!
        update()

        # Log info
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='vpg')
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    vpg(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
