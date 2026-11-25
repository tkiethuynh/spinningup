import importlib
from functools import partial
import os

import gymnasium as gym
import numpy as np
import torch

from spinup import ppo_pytorch
from spinup.exercises.pytorch.problem_set_1 import exercise1_2_auxiliary
from spinup.exercises.pytorch.problem_set_1_solutions import exercise1_1_soln, exercise1_2_soln
from spinup.exercises.pytorch.problem_set_2 import exercise2_2
from spinup.utils.device_utils import get_tf_device, get_torch_device, setup_tf_gpu


def test_pg_math_examples():
    mod1 = importlib.import_module("spinup.examples.pytorch.pg_math.1_simple_pg")
    mod2 = importlib.import_module("spinup.examples.pytorch.pg_math.2_rtg_pg")
    mod1.train(env_name="CartPole-v1", epochs=2, batch_size=200)
    mod2.train(env_name="CartPole-v1", epochs=2, batch_size=200)

    tf_mod1 = importlib.import_module("spinup.examples.tf2.pg_math.1_simple_pg")
    tf_mod2 = importlib.import_module("spinup.examples.tf2.pg_math.2_rtg_pg")
    tf_mod1.train(env_name="CartPole-v1", epochs=2, batch_size=200)
    tf_mod2.train(env_name="CartPole-v1", epochs=2, batch_size=200)


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
