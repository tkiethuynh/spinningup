from functools import partial
import gymnasium as gym
import pandas as pd
import pytest
import torch

from spinup import (
    ddpg_pytorch,
    ppo_pytorch,
    sac_pytorch,
    td3_pytorch,
    trpo_pytorch,
    vpg_pytorch,
)
import spinup.algos.pytorch.ppo.ppo as ppo_mod
import spinup.algos.pytorch.sac.sac as sac_mod


@pytest.mark.parametrize("name,algo", [
    ("vpg", vpg_pytorch),
    ("trpo", trpo_pytorch),
    ("ppo", ppo_pytorch),
])
@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_on_policy_pytorch_algos(tmp_path, name, algo, env_id):
    out_dir = tmp_path / f"{name}_{env_id}"
    algo(
        env_fn=partial(gym.make, env_id),
        ac_kwargs=dict(hidden_sizes=(32,)),
        steps_per_epoch=120,
        epochs=2,
        max_ep_len=50,
        logger_kwargs=dict(output_dir=str(out_dir), exp_name=f"{name}_{env_id}"),
    )
    df = pd.read_table(out_dir / "progress.txt")
    assert len(df) == 2
    assert df["AverageEpRet"].notna().all()
    if name == "trpo":
        assert "KL" in df.columns
        assert "Entropy" in df.columns
        assert (df["KL"] >= 0.0).all()
        assert (df["KL"] <= 1.5 * 0.01 + 1e-4).all()


@pytest.mark.parametrize("name,algo", [
    ("ddpg", ddpg_pytorch),
    ("td3", td3_pytorch),
    ("sac", sac_pytorch),
])
def test_off_policy_pytorch_algos(tmp_path, name, algo):
    out_dir = tmp_path / name
    algo(
        env_fn=partial(gym.make, "Pendulum-v1"),
        ac_kwargs=dict(hidden_sizes=(32, 32)),
        steps_per_epoch=120,
        epochs=2,
        start_steps=40,
        update_after=40,
        update_every=20,
        batch_size=32,
        num_test_episodes=2,
        max_ep_len=50,
        logger_kwargs=dict(output_dir=str(out_dir), exp_name=name),
    )
    df = pd.read_table(out_dir / "progress.txt")
    assert len(df) == 2
    assert df["AverageTestEpRet"].notna().all()
    assert df["TotalEnvInteracts"].iloc[-1] == 240


def test_pytorch_compile_triggers_dynamo(tmp_path):
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    ppo_dir = tmp_path / "ppo_compile"
    ppo_pytorch(
        env_fn=partial(gym.make, "CartPole-v1"),
        ac_kwargs=dict(hidden_sizes=(32,)),
        steps_per_epoch=100,
        epochs=2,
        compile=True,
        logger_kwargs=dict(output_dir=str(ppo_dir), exp_name="ppo_compile"),
    )
    assert torch._dynamo.utils.counters["frames"]["ok"] > 0

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    sac_dir = tmp_path / "sac_compile"
    sac_pytorch(
        env_fn=partial(gym.make, "Pendulum-v1"),
        ac_kwargs=dict(hidden_sizes=(32, 32)),
        steps_per_epoch=80,
        epochs=2,
        start_steps=30,
        update_after=30,
        update_every=20,
        batch_size=16,
        num_test_episodes=1,
        compile=True,
        logger_kwargs=dict(output_dir=str(sac_dir), exp_name="sac_compile"),
    )
    assert torch._dynamo.utils.counters["frames"]["ok"] > 0


def test_pytorch_terminated_vs_truncated_bootstrapping(tmp_path, monkeypatch):
    # Off-policy: Pendulum-v1 truncates at step 200 (< max_ep_len=1000); done_buf must stay 0.0
    recorded_dones = []
    orig_rb = sac_mod.ReplayBuffer

    class SpyReplayBuffer(orig_rb):
        def store(self, obs, act, rew, next_obs, done):
            recorded_dones.append(bool(done))
            super().store(obs, act, rew, next_obs, done)

    monkeypatch.setattr(sac_mod, "ReplayBuffer", SpyReplayBuffer)
    sac_pytorch(
        env_fn=partial(gym.make, "Pendulum-v1"),
        ac_kwargs=dict(hidden_sizes=(32, 32)),
        steps_per_epoch=220,
        epochs=1,
        start_steps=220,
        update_after=9999,
        num_test_episodes=1,
        max_ep_len=1000,
        logger_kwargs=dict(output_dir=str(tmp_path / "sac_trunc"), exp_name="sac_trunc"),
    )
    assert len(recorded_dones) == 220
    assert all(d is False for d in recorded_dones)

    # On-policy: Truncation before max_ep_len must bootstrap non-zero value
    recorded_last_vals = []
    orig_ppo_buf = ppo_mod.PPOBuffer

    class SpyPPOBuffer(orig_ppo_buf):
        def finish_path(self, last_val=0):
            recorded_last_vals.append(float(last_val))
            super().finish_path(last_val)

    monkeypatch.setattr(ppo_mod, "PPOBuffer", SpyPPOBuffer)

    class TruncatingEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype="float32")
            self.action_space = gym.spaces.Discrete(2)
            self.t = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            return self.observation_space.sample() * 0.0 + 1.0, {}

        def step(self, action):
            self.t += 1
            truncated = self.t >= 5
            return self.observation_space.sample() * 0.0 + 1.0, 1.0, False, truncated, {}

    ppo_pytorch(
        env_fn=TruncatingEnv,
        ac_kwargs=dict(hidden_sizes=(32,)),
        steps_per_epoch=10,
        epochs=1,
        max_ep_len=100,
        logger_kwargs=dict(output_dir=str(tmp_path / "ppo_trunc"), exp_name="ppo_trunc"),
    )
    assert len(recorded_last_vals) == 2
    assert all(v != 0.0 for v in recorded_last_vals)
