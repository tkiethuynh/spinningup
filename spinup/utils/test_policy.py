import json
import os
import os.path as osp
import re
import sys
import time
import joblib
import numpy as np

from spinup import EpochLogger
from spinup.utils.device_utils import get_torch_device
from spinup.utils.logx import _extract_env_meta


def _has_display():
    """Return True if a graphical display server appears available."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _reconstruct_env(env=None, env_meta=None, fpath=None, render_mode=None):
    """Recreate a Gymnasium environment from an existing env, env_meta, or config.json."""
    import gymnasium as gym
    from spinup.utils.run_utils import _resolve_env_name

    spec_id = None
    make_kwargs = {}
    flatten_obs = None

    meta = _extract_env_meta(env) if env is not None else None
    if meta is None and isinstance(env_meta, dict) and env_meta.get("id"):
        meta = env_meta

    if meta is not None:
        spec_id = meta.get("id")
        make_kwargs = dict(meta.get("kwargs", {}) or {})
        flatten_obs = meta.get("flatten_obs")
    elif fpath is not None:
        cfg_path = osp.join(fpath, "config.json")
        if osp.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    config = json.load(f)
                if isinstance(config.get("env_name"), str):
                    spec_id = config["env_name"]
                elif "env_fn" in config:
                    m = re.search(
                        r"(?:_make_env|gym\.make)[^'\"]*['\"]([^'\"]+)['\"]",
                        str(config["env_fn"]),
                    )
                    if m:
                        spec_id = m.group(1)
            except Exception:
                pass

    if spec_id is None:
        return env

    if render_mode == "rgb_array" and not _has_display():
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("MUJOCO_GL", "egl")

    resolved_id, resolved_kwargs = _resolve_env_name(spec_id)
    kwargs = {**resolved_kwargs, **make_kwargs}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    new_env = gym.make(resolved_id, **kwargs)
    if flatten_obs is True or (
        flatten_obs is None
        and isinstance(new_env.observation_space, gym.spaces.Box)
        and len(new_env.observation_space.shape) > 1
    ):
        new_env = gym.wrappers.FlattenObservation(new_env)
    return new_env


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

    env = None
    env_meta = None
    try:
        state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
        if isinstance(state, dict):
            env = state.get('env')
            env_meta = state.get('env_meta')
    except Exception:
        pass

    if env is None:
        try:
            env = _reconstruct_env(env_meta=env_meta, fpath=fpath)
        except Exception:
            env = None

    return env, get_action


def load_tf2_policy(fpath, itr, deterministic=False):
    """Load a TF2 policy saved with Spinning Up Logger."""
    from spinup.utils.device_utils import setup_tf_gpu

    setup_tf_gpu()
    import tensorflow as tf
    from spinup.utils.logx import restore_tf2_model

    fname = osp.join(fpath, 'tf2_save', 'model' + itr + '.keras')
    if not osp.exists(fname) and osp.exists(osp.join(fpath, 'tf2_save', 'model' + itr)):
        fname = osp.join(fpath, 'tf2_save', 'model' + itr)
    print('\n\nLoading from %s.\n\n' % fname)

    model = restore_tf2_model(fname)

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
    import torch

    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    device = get_torch_device("cpu")
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


def _with_human_render_mode(env):
    """Re-create a Gymnasium env with render_mode='human' when a display is available."""
    if getattr(env, 'render_mode', None) is not None or getattr(env, 'spec', None) is None:
        return env
    if not _has_display():
        return env
    try:
        return _reconstruct_env(env=env, render_mode='human')
    except Exception:
        return env


def _save_episode_gif(frames, out_path, fps=30):
    """Save a sequence of RGB uint8 numpy frames as an animated GIF via Pillow."""
    from PIL import Image

    if not frames:
        return
    pil_frames = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    duration_ms = max(1, int(round(1000.0 / max(fps, 1))))
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, save_video=None):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    if max_ep_len is not None and max_ep_len <= 0:
        max_ep_len = None

    active_env = env
    live_render = False
    record_video = bool(save_video)

    if record_video:
        os.makedirs(save_video, exist_ok=True)
        if getattr(env, 'render_mode', None) != 'rgb_array':
            try:
                active_env = _reconstruct_env(env=env, render_mode='rgb_array')
            except Exception as exc:
                print(
                    f"Warning: Could not initialize rgb_array rendering ({exc}). "
                    "On headless CPU-only nodes without EGL/OSMesa, MuJoCo video capture "
                    "is unavailable; disabling video recording."
                )
                active_env = env
                record_video = False
    elif render:
        if not _has_display():
            print(
                "Warning: No display detected ($DISPLAY / $WAYLAND_DISPLAY unset, e.g. headless SSH session); "
                "skipping live window rendering. Run from a desktop terminal, pass --norender, "
                "or pass --save_video <dir> to save episode GIFs."
            )
            live_render = False
        else:
            active_env = _with_human_render_mode(env)
            live_render = getattr(active_env, 'render_mode', None) is not None

    logger = EpochLogger()
    n = 0
    try:
        o, _ = active_env.reset()
        frames = []
        if record_video:
            try:
                frame = active_env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception as exc:
                print(f"Warning: rgb_array render failed ({exc}); disabling video recording.")
                record_video = False
                frames = []

        r, d, ep_ret, ep_len = 0, False, 0, 0
        fps = int((getattr(active_env, 'metadata', {}) or {}).get('render_fps') or 30)

        while n < num_episodes:
            if live_render:
                active_env.render()
                time.sleep(1e-3)

            a = get_action(o)
            o, r, terminated, truncated, _ = active_env.step(a)
            d = terminated or truncated
            ep_ret += r
            ep_len += 1

            if record_video:
                try:
                    frame = active_env.render()
                    if frame is not None:
                        frames.append(frame)
                except Exception as exc:
                    print(f"Warning: rgb_array render failed ({exc}); disabling video recording.")
                    record_video = False
                    frames = []

            if d or (ep_len == max_ep_len):
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
                if record_video and frames:
                    gif_path = osp.join(save_video, f'episode_{n}.gif')
                    _save_episode_gif(frames, gif_path, fps=fps)
                    print('Saved video to %s' % gif_path)
                n += 1
                if n < num_episodes:
                    o, _ = active_env.reset()
                    frames = []
                    if record_video:
                        frame = active_env.render()
                        if frame is not None:
                            frames.append(frame)
                r, d, ep_ret, ep_len = 0, False, 0, 0
    except KeyboardInterrupt:
        print('\nStopped early by user.')
    finally:
        try:
            active_env.close()
        except Exception:
            pass
        if active_env is not env:
            try:
                env.close()
            except Exception:
                pass

    if n > 0:
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=None)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--save_video', type=str, default=None)
    args = parser.parse_args()
    env, get_action = load_policy_and_env(
        args.fpath,
        args.itr if args.itr >= 0 else 'last',
        args.deterministic,
    )
    episodes = args.episodes if args.episodes is not None else (3 if args.save_video else 100)
    max_ep_len = args.len if args.len > 0 else None
    run_policy(
        env,
        get_action,
        max_ep_len,
        episodes,
        not args.norender,
        save_video=args.save_video,
    )
