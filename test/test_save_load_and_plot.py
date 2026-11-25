from functools import partial
import warnings
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from spinup import ppo_pytorch, ppo_tf2, sac_pytorch, sac_tf2
from spinup.utils.logx import restore_tf2_model
from spinup.utils import plot as plot_mod
from spinup.utils.test_policy import _with_human_render_mode, load_policy_and_env, run_policy


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


def test_test_policy_render_mode_upgrade(monkeypatch):
    env = gym.make("CartPole-v1")
    assert env.render_mode is None

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert _with_human_render_mode(env) is env

    monkeypatch.setattr("spinup.utils.test_policy._has_display", lambda: True)
    rendered_env = _with_human_render_mode(env)
    try:
        assert rendered_env.render_mode == "human"
        obs, _ = rendered_env.reset(seed=0)
        run_policy(env, lambda o: env.action_space.sample(), max_ep_len=5, num_episodes=1, render=False)
        assert obs.shape == env.observation_space.shape
    finally:
        env.close()
        rendered_env.close()


def test_unpicklable_flattened_env_save_and_test_policy(tmp_path):
    out_dir = tmp_path / "flat_ppo"
    ppo_pytorch(
        env_fn=lambda: gym.wrappers.FlattenObservation(gym.make("CartPole-v1")),
        ac_kwargs=dict(hidden_sizes=(16,)),
        steps_per_epoch=40,
        epochs=1,
        max_ep_len=20,
        logger_kwargs=dict(output_dir=str(out_dir), exp_name="flat_ppo"),
    )
    assert not (out_dir / "vars.pkl.tmp").exists()
    assert (out_dir / "vars.pkl").exists()

    env, get_action = load_policy_and_env(str(out_dir), deterministic=True)
    assert env is not None
    run_policy(env, get_action, max_ep_len=5, num_episodes=1, render=False)


def test_test_policy_headless_and_save_video(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for ename in ("CartPole-v1", "HalfCheetah-v5"):
            env = gym.make(ename)
            run_policy(env, lambda o, e=env: e.action_space.sample(), max_ep_len=5, num_episodes=1, render=True)

    vid_dir = tmp_path / "videos"
    env_v = gym.make("CartPole-v1")
    run_policy(
        env_v,
        lambda o: env_v.action_space.sample(),
        max_ep_len=5,
        num_episodes=1,
        render=True,
        save_video=str(vid_dir),
    )
    gif_path = vid_dir / "episode_0.gif"
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_plot_edge_cases_nan_smoothing_and_savefig(tmp_path):
    d = tmp_path / "exp" / "exp_s0"
    d.mkdir(parents=True)
    (d / "config.json").write_text('{"exp_name": "exp"}')
    (d / "progress.txt").write_text(
        "Epoch\tTotalEnvInteracts\tAverageEpRet\n"
        "0\t4000\tnan\n"
        "1\t8000\t20.0\n"
        "2\t12000\t40.0\n"
    )
    (tmp_path / "exp_notes.txt").write_text("unrelated file with prefix")

    plot_mod.exp_idx = 0
    plot_mod.units = {}
    datasets = plot_mod.get_all_datasets([str(tmp_path / "exp")], legend=["MyExp"])
    assert len(datasets) == 1

    plt.figure()
    plot_mod.plot_data(datasets, xaxis="TotalEnvInteracts", value="Performance", smooth=3, estimator=np.mean)
    assert np.isfinite(datasets[0]["Performance"]).all()
    plt.close("all")

    out_png = tmp_path / "plots" / "out.png"
    plot_mod.exp_idx = 0
    plot_mod.units = {}
    plot_mod.make_plots(
        [str(tmp_path / "exp")],
        legend=["MyExp"],
        xaxis="TotalEnvInteracts",
        values=["Performance"],
        smooth=3,
        savefig=str(out_png),
    )
    assert out_png.exists()
    assert out_png.stat().st_size > 0
    plt.close("all")

    with pytest.raises(ValueError):
        plot_mod.make_plots([str(tmp_path / "missing")])

    with pytest.raises(ValueError):
        plot_mod.make_plots([])
