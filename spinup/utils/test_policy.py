import os
import os.path as osp
import time
import joblib
import numpy as np
import tensorflow as tf
import torch

from spinup import EpochLogger
from spinup.utils.device_utils import get_torch_device


def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF2 or PyTorch, along with RL env.

    Checks to see if there's a tf2_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """
    if any(['tf2_save' in x for x in os.listdir(fpath)]):
        backend = 'tf2'
    else:
        backend = 'pytorch'

    if itr == 'last':
        if backend == 'tf2':
            tf2save_path = osp.join(fpath, 'tf2_save')
            saves = [
                int(x[5:-6])
                for x in os.listdir(tf2save_path)
                if x.startswith('model') and x.endswith('.keras') and len(x) > 11 and x[5:-6].isdigit()
            ]
        else:
            pytsave_path = osp.join(fpath, 'pyt_save')
            saves = [
                int(x.split('.')[0][5:])
                for x in os.listdir(pytsave_path)
                if x.startswith('model') and x.endswith('.pt') and len(x) > 8 and x.split('.')[0][5:].isdigit()
            ]

        itr = '%d' % max(saves) if len(saves) > 0 else ''
    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d' % itr

    if backend == 'tf2':
        get_action = load_tf2_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic)

    try:
        state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
        env = state['env']
    except Exception:
        env = None

    return env, get_action


def load_tf2_policy(fpath, itr, deterministic=False):
    """Load a TF2 policy saved with Spinning Up Logger."""
    fname = osp.join(fpath, 'tf2_save', 'model' + itr + '.keras')
    if not osp.exists(fname) and osp.exists(osp.join(fpath, 'tf2_save', 'model' + itr)):
        fname = osp.join(fpath, 'tf2_save', 'model' + itr)
    print('\n\nLoading from %s.\n\n' % fname)

    import spinup  # noqa: F401 - ensure custom Keras classes are registered
    model = tf.keras.models.load_model(fname, compile=False)

    def get_action(x):
        obs = tf.convert_to_tensor(np.asarray(x, dtype=np.float32).reshape(1, -1))
        if hasattr(model, 'act'):
            try:
                return model.act(obs, deterministic=deterministic)[0]
            except TypeError:
                return model.act(obs)[0]
        else:
            out = model.pi(obs)
            a = out[0] if isinstance(out, (tuple, list)) else out
            return a.numpy()[0]

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """Load a PyTorch policy saved with Spinning Up Logger."""
    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    device = get_torch_device()
    model = torch.load(fname, map_location=device, weights_only=False)
    model.eval()

    def get_action(x):
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
            try:
                action = model.act(x_t, deterministic=deterministic)
            except TypeError:
                action = model.act(x_t)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, _ = env.reset()
    r, d, ep_ret, ep_len, n = 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, terminated, truncated, _ = env.step(a)
        d = terminated or truncated
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            o, _ = env.reset()
            r, d, ep_ret, ep_len = 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(
        args.fpath,
        args.itr if args.itr >= 0 else 'last',
        args.deterministic,
    )
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))
