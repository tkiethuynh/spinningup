from functools import partial
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spinup import ppo_pytorch, ppo_tf2, sac_pytorch, sac_tf2
from spinup.utils.logx import restore_tf2_model
from spinup.utils import plot as plot_mod
from spinup.utils.test_policy import load_policy_and_env


def test_save_load_pytorch_roundtrip(tmp_path):
    # Discrete PPO
    ppo_dir = tmp_path / "ppo_pyt"
    ppo_pytorch(
        env_fn=partial(gym.make, "CartPole-v1"),
        ac_kwargs=dict(hidden_sizes=(32,)),
        steps_per_epoch=80,
        epochs=2,
        logger_kwargs=dict(output_dir=str(ppo_dir), exp_name="ppo_pyt"),
    )
    env, get_act_det = load_policy_and_env(str(ppo_dir), deterministic=True)
    obs, _ = env.reset(seed=0)
    a1 = get_act_det(obs)
    a2 = get_act_det(obs)
    assert int(a1) == int(a2)
    assert env.action_space.contains(int(a1))

    # Continuous SAC deterministic forwarding check
    sac_dir = tmp_path / "sac_pyt"
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
        logger_kwargs=dict(output_dir=str(sac_dir), exp_name="sac_pyt"),
    )
    env_c, get_sac_det = load_policy_and_env(str(sac_dir), deterministic=True)
    oc, _ = env_c.reset(seed=0)
    ac1 = get_sac_det(oc)
    ac2 = get_sac_det(oc)
    assert np.allclose(ac1, ac2)
    assert env_c.action_space.contains(np.asarray(ac1, dtype=np.float32))


def test_save_load_tf2_roundtrip(tmp_path):
    # Discrete PPO TF2
    ppo_d_dir = tmp_path / "ppo_tf2_d"
    ppo_tf2(
        env_fn=partial(gym.make, "CartPole-v1"),
        ac_kwargs=dict(hidden_sizes=(32,)),
        steps_per_epoch=80,
        epochs=2,
        logger_kwargs=dict(output_dir=str(ppo_d_dir), exp_name="ppo_tf2_d"),
    )
    assert (ppo_d_dir / "tf2_save" / "model.keras").exists()
    env_d, get_d = load_policy_and_env(str(ppo_d_dir), deterministic=True)
    od, _ = env_d.reset(seed=0)
    assert int(get_d(od)) == int(get_d(od))
    assert env_d.action_space.contains(int(get_d(od)))
    loaded = restore_tf2_model(str(ppo_d_dir / "tf2_save" / "model.keras"))
    assert hasattr(loaded, "act")

    # Continuous PPO TF2
    ppo_c_dir = tmp_path / "ppo_tf2_c"
    ppo_tf2(
        env_fn=partial(gym.make, "Pendulum-v1"),
        ac_kwargs=dict(hidden_sizes=(32,)),
        steps_per_epoch=80,
        epochs=2,
        logger_kwargs=dict(output_dir=str(ppo_c_dir), exp_name="ppo_tf2_c"),
    )
    env_c, get_c = load_policy_and_env(str(ppo_c_dir), deterministic=True)
    oc, _ = env_c.reset(seed=0)
    assert np.allclose(get_c(oc), get_c(oc))

    # Continuous SAC TF2
    sac_dir = tmp_path / "sac_tf2"
    sac_tf2(
        env_fn=partial(gym.make, "Pendulum-v1"),
        ac_kwargs=dict(hidden_sizes=(32, 32)),
        steps_per_epoch=80,
        epochs=2,
        start_steps=30,
        update_after=30,
        update_every=20,
        batch_size=16,
        num_test_episodes=1,
        logger_kwargs=dict(output_dir=str(sac_dir), exp_name="sac_tf2"),
    )
    env_s, get_s = load_policy_and_env(str(sac_dir), deterministic=True)
    os_, _ = env_s.reset(seed=0)
    as1 = get_s(os_)
    as2 = get_s(os_)
    assert np.allclose(as1, as2)


def test_plot_data_and_make_plots(tmp_path, monkeypatch):
    for seed in (0, 1):
        out_dir = tmp_path / "exp" / f"exp_s{seed}"
        ppo_pytorch(
            env_fn=partial(gym.make, "CartPole-v1"),
            ac_kwargs=dict(hidden_sizes=(16,)),
            seed=seed,
            steps_per_epoch=80,
            epochs=2,
            logger_kwargs=dict(output_dir=str(out_dir), exp_name="exp_plot"),
        )

    monkeypatch.setattr(plt, "show", lambda: None)
    plot_mod.exp_idx = 0
    plot_mod.units = {}
    plot_mod.make_plots(
        [str(tmp_path / "exp") + "/"],
        xaxis="TotalEnvInteracts",
        values=["Performance"],
    )
    assert len(plt.get_fignums()) >= 1
    plt.close("all")
