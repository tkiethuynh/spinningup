import importlib
from functools import partial
import os
import subprocess
import sys
import warnings

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch

from spinup import ppo_pytorch, ppo_tf2, sac_pytorch, sac_tf2
from spinup.exercises.pytorch.problem_set_1 import exercise1_2_auxiliary
from spinup.exercises.pytorch.problem_set_1_solutions import exercise1_1_soln, exercise1_2_soln
from spinup.exercises.pytorch.problem_set_2 import exercise2_2
from spinup.utils import mpi_tools
from spinup.utils.device_utils import get_tf_device, get_torch_device, setup_tf_gpu
from spinup.utils.run_utils import _make_env
from spinup.utils.test_policy import load_policy_and_env


def test_pg_math_examples(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    mod1 = importlib.import_module("spinup.examples.pytorch.pg_math.1_simple_pg")
    mod2 = importlib.import_module("spinup.examples.pytorch.pg_math.2_rtg_pg")
    mod1.train(env_name="CartPole-v1", epochs=2, batch_size=200)
    mod2.train(env_name="CartPole-v1", epochs=2, batch_size=200)

    tf_mod1 = importlib.import_module("spinup.examples.tf2.pg_math.1_simple_pg")
    tf_mod2 = importlib.import_module("spinup.examples.tf2.pg_math.2_rtg_pg")
    tf_mod1.train(env_name="CartPole-v1", epochs=2, batch_size=200)
    tf_mod2.train(env_name="CartPole-v1", epochs=2, batch_size=200)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mod1.train(env_name="CartPole-v1", epochs=1, batch_size=50, render=True)
        tf_mod1.train(env_name="CartPole-v1", epochs=1, batch_size=50, render=True)


def test_run_cli_parsing_edge_cases(tmp_path):
    for exp_name in ("dir", "123"):
        data_dir = tmp_path / "my--runs"
        cli_args = [
            "ppo",
            "--env", "CartPole-v1",
            "--exp_name", exp_name,
            "--hid", "[16]",
            "--epochs", "1",
            "--steps", "40",
            "--max_ep_len", "20",
            "--data_dir", str(data_dir),
        ]
        script = (
            "import sys, spinup.user_config as uc; uc.WAIT_BEFORE_LAUNCH = 0; "
            f"sys.argv = ['spinup/run.py'] + {cli_args!r}; "
            "import runpy; runpy.run_module('spinup.run', run_name='__main__')"
        )
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        assert (data_dir / exp_name / f"{exp_name}_s0" / "progress.txt").exists()


def test_exercise_solutions_and_bugged(tmp_path):
    x = torch.randn(16, 4)
    mu = torch.randn(16, 4)
    log_std = torch.randn(4)
    expected = torch.distributions.Normal(mu, torch.exp(log_std)).log_prob(x).sum(dim=-1)
    actual = exercise1_1_soln.gaussian_likelihood(x, mu, log_std)
    assert torch.allclose(actual, expected, atol=1e-5)

    actor_critic = partial(
        exercise1_2_auxiliary.ExerciseActorCritic,
        actor=exercise1_2_soln.MLPGaussianActor,
    )
    ppo_pytorch(
        env_fn=partial(gym.make, "Pendulum-v1"),
        actor_critic=actor_critic,
        ac_kwargs=dict(hidden_sizes=(32,)),
        steps_per_epoch=100,
        epochs=2,
        logger_kwargs=dict(output_dir=str(tmp_path / "ex12"), exp_name="ex12"),
    )

    env = gym.make("Pendulum-v1")
    bugged = exercise2_2.BuggedMLPActorCritic(env.observation_space, env.action_space, hidden_sizes=(32,))
    act = bugged.act(torch.zeros(env.observation_space.shape[0], dtype=torch.float32))
    assert isinstance(act, np.ndarray)


def test_device_utils_cleanliness():
    assert not os.path.exists(os.path.expanduser("~/tf_gpu_libs"))
    setup_tf_gpu()
    assert get_torch_device().type in ("cpu", "cuda", "mps")
    assert get_tf_device() in ("/GPU:0", "/device:CPU:0")


def test_box2d_lunarlander_smoke(tmp_path):
    pytest.importorskip("Box2D")
    out_dir = tmp_path / "lunarlander"
    ppo_pytorch(
        env_fn=partial(gym.make, "LunarLander-v3"),
        ac_kwargs=dict(hidden_sizes=(32,)),
        steps_per_epoch=80,
        epochs=1,
        max_ep_len=40,
        logger_kwargs=dict(output_dir=str(out_dir), exp_name="lunarlander"),
    )
    assert (out_dir / "progress.txt").exists()


def test_multi_gpu_mpi_device_mapping(monkeypatch):
    setup_tf_gpu()
    import tensorflow as tf

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(mpi_tools, "num_procs", lambda: 4)
    monkeypatch.setattr(
        tf.config,
        "list_physical_devices",
        lambda dev_type=None: [object()] * 4 if dev_type == "GPU" else [],
    )

    for rank in (0, 1, 2, 3, 5):
        monkeypatch.setattr(mpi_tools, "proc_id", lambda r=rank: r)
        expected_idx = rank % 4
        assert get_torch_device("auto") == torch.device(f"cuda:{expected_idx}")
        assert get_torch_device("gpu") == torch.device(f"cuda:{expected_idx}")
        assert get_tf_device("auto") == f"/GPU:{expected_idx}"
        assert get_tf_device("cuda") == f"/GPU:{expected_idx}"
        assert get_tf_device("gpu") == f"/GPU:{expected_idx}"

    assert get_torch_device("gpu:2") == torch.device("cuda:2")
    assert get_torch_device("/GPU:1") == torch.device("cuda:1")
    assert get_torch_device("cpu") == torch.device("cpu")
    assert get_tf_device("cuda:2") == "/GPU:2"
    assert get_tf_device("/GPU:1") == "/GPU:1"
    assert get_tf_device("cpu") == "/device:CPU:0"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda dev_type=None: [])
    assert get_torch_device("auto") == torch.device("cpu")
    assert get_torch_device("cuda:2") == torch.device("cpu")
    assert get_tf_device("auto") == "/device:CPU:0"
    assert get_tf_device("gpu:1") == "/device:CPU:0"


def test_atari_pong_smoke(tmp_path):
    ale_py = pytest.importorskip("ale_py")
    gym.register_envs(ale_py)

    for name, algo in [("ppo_pyt", ppo_pytorch), ("ppo_tf2", ppo_tf2)]:
        out_dir = tmp_path / name
        algo(
            env_fn=partial(gym.make, "ALE/Pong-v5", obs_type="ram"),
            ac_kwargs=dict(hidden_sizes=(32,)),
            steps_per_epoch=80,
            epochs=1,
            max_ep_len=40,
            logger_kwargs=dict(output_dir=str(out_dir), exp_name=name),
        )
        df = pd.read_table(out_dir / "progress.txt")
        assert len(df) == 1
        assert df["AverageEpRet"].notna().all()

    env_ram = _make_env("ALE/Pong-ram-v5")
    obs_ram, _ = env_ram.reset(seed=0)
    assert obs_ram.shape == (128,)
    env_ram.close()

    env_px = _make_env("ALE/Pong-v5")
    obs, _ = env_px.reset(seed=0)
    assert obs.ndim == 1
    next_obs, _, _, _, _ = env_px.step(env_px.action_space.sample())
    assert next_obs.shape == obs.shape
    env_px.close()

    # Flattened pixel Atari env must save env_meta fallback cleanly and reload via test_policy
    px_dir = tmp_path / "pong_px"
    ppo_pytorch(
        env_fn=partial(_make_env, "ALE/Pong-v5"),
        ac_kwargs=dict(hidden_sizes=(16,)),
        steps_per_epoch=40,
        epochs=1,
        max_ep_len=20,
        logger_kwargs=dict(output_dir=str(px_dir), exp_name="pong_px"),
    )
    loaded_env, get_act = load_policy_and_env(str(px_dir), deterministic=True)
    assert loaded_env is not None
    o_px, _ = loaded_env.reset(seed=0)
    a_px = get_act(o_px)
    loaded_env.step(a_px)
    loaded_env.close()


def test_mujoco_v5_algos_and_convergence(tmp_path):
    on_specs = [
        ("ppo_pyt", ppo_pytorch, "pyt_save/model.pt"),
        ("ppo_tf2", ppo_tf2, "tf2_save/model.keras"),
    ]
    for name, algo, ckpt in on_specs:
        out_dir = tmp_path / f"hc_{name}"
        algo(
            env_fn=partial(gym.make, "HalfCheetah-v5"),
            ac_kwargs=dict(hidden_sizes=(32,)),
            steps_per_epoch=100,
            epochs=1,
            max_ep_len=50,
            logger_kwargs=dict(output_dir=str(out_dir), exp_name=name),
        )
        df = pd.read_table(out_dir / "progress.txt")
        assert len(df) == 1 and df["AverageEpRet"].notna().all()
        assert (out_dir / ckpt).exists()

    off_specs = [
        ("sac_pyt", sac_pytorch, "pyt_save/model.pt"),
        ("sac_tf2", sac_tf2, "tf2_save/model.keras"),
    ]
    for name, algo, ckpt in off_specs:
        out_dir = tmp_path / f"hc_{name}"
        algo(
            env_fn=partial(gym.make, "HalfCheetah-v5"),
            ac_kwargs=dict(hidden_sizes=(32, 32)),
            steps_per_epoch=100,
            epochs=1,
            start_steps=30,
            update_after=30,
            update_every=20,
            batch_size=32,
            num_test_episodes=1,
            max_ep_len=50,
            logger_kwargs=dict(output_dir=str(out_dir), exp_name=name),
        )
        df = pd.read_table(out_dir / "progress.txt")
        assert len(df) == 1 and df["AverageTestEpRet"].notna().all()
        assert (out_dir / ckpt).exists()

    # Seeded convergence checks on MuJoCo InvertedPendulum-v5
    ppo_conv_dir = tmp_path / "ip_ppo_conv"
    ppo_pytorch(
        env_fn=partial(gym.make, "InvertedPendulum-v5"),
        ac_kwargs=dict(hidden_sizes=(32, 32)),
        seed=0,
        steps_per_epoch=500,
        epochs=8,
        max_ep_len=200,
        pi_lr=1e-3,
        vf_lr=3e-3,
        logger_kwargs=dict(output_dir=str(ppo_conv_dir), exp_name="ip_ppo_conv"),
    )
    df_ppo = pd.read_table(ppo_conv_dir / "progress.txt")
    assert df_ppo["AverageEpRet"].iloc[-1] > 25.0
    assert df_ppo["AverageEpRet"].iloc[-1] > 2.0 * df_ppo["AverageEpRet"].iloc[0]

    sac_conv_dir = tmp_path / "ip_sac_conv"
    sac_pytorch(
        env_fn=partial(gym.make, "InvertedPendulum-v5"),
        ac_kwargs=dict(hidden_sizes=(64, 64)),
        seed=0,
        steps_per_epoch=400,
        epochs=4,
        start_steps=200,
        update_after=200,
        update_every=50,
        batch_size=64,
        num_test_episodes=3,
        max_ep_len=200,
        logger_kwargs=dict(output_dir=str(sac_conv_dir), exp_name="ip_sac_conv"),
    )
    df_sac = pd.read_table(sac_conv_dir / "progress.txt")
    assert df_sac["AverageTestEpRet"].iloc[0] < 25.0
    assert df_sac["AverageTestEpRet"].max() >= 50.0
