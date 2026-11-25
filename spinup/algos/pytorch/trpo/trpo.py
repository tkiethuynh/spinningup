import time
from typing import Callable, Dict, Any

import gymnasium as gym
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam

import spinup.algos.pytorch.trpo.core as core
from spinup.utils.device_utils import get_torch_device
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class TRPOBuffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim: tuple, act_dim: tuple, size: int, gamma: float = 0.99, lam: float = 0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float):
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

    def finish_path(self, last_val: float = 0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
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

    def get(self, device: torch.device):
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
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}


def trpo(env_fn: Callable[[], gym.Env],
         actor_critic: type[torch.nn.Module] = core.MLPActorCritic,
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
    Trust Region Policy Optimization

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the Gymnasium API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

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

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
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

        device (str): Compute device (`'cpu'`, `'cuda'`, `'mps'`, or `'auto'`).
            Defaults to `'cpu'`, which is usually fastest for single-environment
            rollouts with shallow MLPs; use `'auto'` or `'cuda'` for larger
            networks or GPU setups.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Select compute device
    torch_device = get_torch_device(device)
    logger.log(f'Using device: {torch_device}')

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(torch_device)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = TRPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    def _detached_dist(obs):
        # Snapshot old policy distribution pi_{theta_k} (held fixed during HVP and line search)
        with torch.no_grad():
            pi_dist, _ = ac.pi(obs)
            if isinstance(pi_dist, Categorical):
                return Categorical(logits=pi_dist.logits.detach())
            if isinstance(pi_dist, Normal):
                return Normal(pi_dist.loc.detach(), pi_dist.scale.detach())
            return pi_dist

    # Set up function for computing TRPO surrogate policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        ent = pi.entropy().mean()
        return loss_pi, ent

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up optimizer for value function (policy is updated by natural gradient step)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def cg(hvp_fn, b):
        """
        Conjugate gradient algorithm (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        for approximately solving Hx = b without forming the full Fisher matrix H.
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
        data = buf.get(torch_device)
        obs = data['obs']
        pi_params = list(ac.pi.parameters())
        pi_old = _detached_dist(obs)

        # Evaluate pre-update surrogate loss, entropy, and value loss
        loss_pi_cur, ent_cur = compute_loss_pi(data)
        loss_pi_old = float(mpi_avg(loss_pi_cur.item()))
        ent_val = float(mpi_avg(ent_cur.item()))
        v_l_old = compute_loss_v(data).item()

        # Policy gradient g (averaged across MPI ranks)
        g_local = core.flat_grad(loss_pi_cur, pi_params).detach().cpu().numpy()
        g = mpi_avg(g_local)

        def hvp(v_np):
            # Damped Fisher-vector product: H v = grad_theta (grad_theta KL(pi_old || pi) . v) + alpha v
            v_t = torch.as_tensor(v_np, dtype=torch.float32, device=torch_device)
            pi_new, _ = ac.pi(obs)
            kl = core.kl_divergence(pi_old, pi_new)
            kl_grad = core.flat_grad(kl, pi_params, retain_graph=True, create_graph=True)
            grad_v = (kl_grad * v_t).sum()
            hvp_t = core.flat_grad(grad_v, pi_params) + damping_coeff * v_t
            return mpi_avg(hvp_t.detach().cpu().numpy())

        # Solve H x ~= g via conjugate gradients and compute max step length for KL <= delta
        x = cg(hvp, g)
        alpha_val = np.sqrt(2 * delta / (np.dot(x, hvp(x)) + 1e-8))
        old_params = core.flat_params(pi_params)

        # Backtracking line search: accept first step satisfying KL limit and loss improvement
        accepted_iter = backtrack_iters
        for j in range(backtrack_iters):
            step_t = torch.as_tensor(
                alpha_val * x * (backtrack_coeff ** j), dtype=torch.float32, device=torch_device
            )
            core.set_flat_params(pi_params, old_params - step_t)
            with torch.no_grad():
                pi_new, _ = ac.pi(obs)
                kl_cand = float(mpi_avg(core.kl_divergence(pi_old, pi_new).item()))
                loss_cand = float(mpi_avg(compute_loss_pi(data)[0].item()))
            if kl_cand <= 1.5 * delta and loss_cand <= loss_pi_old:
                accepted_iter = j
                break
        else:
            # Revert to old parameters if no backtracking candidate satisfied constraints
            core.set_flat_params(pi_params, old_params)

        # Value function learning
        for _ in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Evaluate post-update diagnostics
        with torch.no_grad():
            pi_new, _ = ac.pi(obs)
            kl_final = core.kl_divergence(pi_old, pi_new).item()
            loss_pi_new = compute_loss_pi(data)[0].item()
            loss_v_new = compute_loss_v(data).item()

        logger.store(
            LossPi=loss_pi_old,
            LossV=v_l_old,
            KL=kl_final,
            Entropy=ent_val,
            DeltaLossPi=(loss_pi_new - loss_pi_old),
            DeltaLossV=(loss_v_new - v_l_old),
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
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32, device=torch_device))

            next_o, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = (ep_len == max_ep_len) or truncated
            terminal = terminated or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                # Only zero bootstrap on true termination; bootstrap on truncation/horizon/epoch cutoff
                if terminated:
                    v = 0.0
                else:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32, device=torch_device))
                buf.finish_path(v)
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

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    trpo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs, device=args.device)
