from copy import deepcopy
import itertools
import time
from typing import Callable, Optional, Dict, Any

import gymnasium as gym
import numpy as np
import torch
from torch.optim import Adam

import spinup.algos.pytorch.sac.core as core
from spinup.utils.device_utils import get_torch_device
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim: tuple, act_dim: int, size: int):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int = 32, device: Optional[torch.device] = None):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}


def sac(env_fn: Callable[[], gym.Env],
        actor_critic: type[torch.nn.Module] = core.MLPActorCritic,
        ac_kwargs: Dict[str, Any] = dict(),
        seed: int = 0,
        steps_per_epoch: int = 4000,
        epochs: int = 100,
        replay_size: int = int(1e6),
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
        save_freq: int = 1,
        compile: bool = False):
    """
    Soft Actor-Critic (SAC)
    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Device selection
    device = get_torch_device()
    logger.log(f'Using device: {device}')

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac).to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    # Pure-tensor loss kernels (graph-friendly for torch.compile)
    def _loss_q_tensors(o, a, r, o2, d):
        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)
        with torch.no_grad():
            a2, logp_a2 = ac.pi(o2)
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2
        return loss_q, q1.detach(), q2.detach()

    def _loss_pi_tensors(o):
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (alpha * logp_pi - q_pi).mean()
        return loss_pi, logp_pi.detach()

    loss_q_fn = _loss_q_tensors
    loss_pi_fn = _loss_pi_tensors
    if compile and hasattr(torch, 'compile'):
        logger.log('Compiling loss computation graphs...')
        loss_q_fn = torch.compile(_loss_q_tensors)
        loss_pi_fn = torch.compile(_loss_pi_tensors)

    def compute_loss_q(data):
        loss_q, q1_det, q2_det = loss_q_fn(
            data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        )
        q_info = dict(Q1Vals=q1_det.cpu().numpy(), Q2Vals=q2_det.cpu().numpy())
        return loss_q, q_info

    def compute_loss_pi(data):
        loss_pi, logp_det = loss_pi_fn(data['obs'])
        pi_info = dict(LogPi=logp_det.cpu().numpy())
        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        logger.store(LossQ=loss_q.item(), **q_info)

        for p in q_params:
            p.requires_grad = False

        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        for p in q_params:
            p.requires_grad = True

        logger.store(LossPi=loss_pi.item(), **pi_info)

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.copy_(polyak * p_targ + (1 - polyak) * p)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32, device=device),
                      deterministic)

    def test_agent():
        for _ in range(num_test_episodes):
            o, _ = test_env.reset()
            d, ep_ret, ep_len = False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                o, r, terminated, truncated, _ = test_env.step(get_action(o, True))
                d = terminated or truncated
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    test_env.reset(seed=seed + 10000)
    test_env.action_space.seed(seed + 10000)
    ep_ret, ep_len = 0, 0

    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        o2, r, terminated, truncated, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Store true termination only; never zero bootstrap on truncation or time limit
        replay_buffer.store(o, a, r, o2, terminated)
        o = o2

        if terminated or truncated or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _ = env.reset()
            ep_ret, ep_len = 0, 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size, device=device)
                update(data=batch)

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            test_agent()

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t + 1)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
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
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, compile=args.compile)
