from functools import partial
import os
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pandas as pd

from spinup import ppo_pytorch, ppo_tf2
from spinup.utils.logx import EpochLogger


def test_mpi_tools_disables_hwloc_gl():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; os.environ.pop('HWLOC_COMPONENTS', None); "
            "import spinup.utils.mpi_tools; "
            "assert os.environ.get('HWLOC_COMPONENTS') == '-gl'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_epoch_logger_handles_empty_first_epoch(tmp_path):
    logger = EpochLogger(output_dir=str(tmp_path), exp_name="empty_test")

    # Epoch 0: no finished episodes stored for EpRet
    logger.log_tabular("Epoch", 0)
    logger.log_tabular("EpRet", with_min_and_max=True)
    logger.dump_tabular()

    # Epoch 1: values arrive; must not fail with late-key AssertionError
    logger.store(EpRet=10.0)
    logger.store(EpRet=20.0)
    logger.log_tabular("Epoch", 1)
    logger.log_tabular("EpRet", with_min_and_max=True)
    logger.dump_tabular()
    logger.output_file.close()

    df = pd.read_table(tmp_path / "progress.txt")
    assert list(df.columns) == ["Epoch", "AverageEpRet", "StdEpRet", "MaxEpRet", "MinEpRet"]
    assert np.isnan(df["AverageEpRet"].iloc[0])
    assert df["AverageEpRet"].iloc[1] == 15.0


def test_deterministic_seeding_pytorch_and_tf2(tmp_path):
    for backend_name, algo in [("pyt", ppo_pytorch), ("tf2", ppo_tf2)]:
        runs = []
        for idx in (0, 1):
            out_dir = tmp_path / f"{backend_name}_run{idx}"
            algo(
                env_fn=partial(gym.make, "CartPole-v1"),
                ac_kwargs=dict(hidden_sizes=(32,)),
                seed=42,
                steps_per_epoch=100,
                epochs=2,
                logger_kwargs=dict(output_dir=str(out_dir), exp_name=f"det_{backend_name}"),
            )
            runs.append(pd.read_table(out_dir / "progress.txt"))
        assert np.allclose(runs[0]["AverageEpRet"], runs[1]["AverageEpRet"])


def test_mpi_sync_and_algos(tmp_path):
    script = f"""
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from spinup.utils.mpi_tools import proc_id, num_procs
from spinup.utils.mpi_pytorch import sync_params, mpi_avg_grads
from spinup.utils.logx import EpochLogger
from spinup import ppo_pytorch, trpo_pytorch, ppo_tf2, trpo_tf2

assert num_procs() == 2

# 1. PyTorch MPI sync_params & mpi_avg_grads
m = nn.Linear(4, 2)
with torch.no_grad():
    m.weight.fill_(float(proc_id() + 1))
sync_params(m)
assert torch.allclose(m.weight, torch.ones_like(m.weight))

loss = (m(torch.ones(2, 4)) * (proc_id() + 1)).sum()
loss.backward()
mpi_avg_grads(m)
# Gradient avg across rank 0 (1x) and rank 1 (2x) == 1.5 * base_grad
assert torch.allclose(m.bias.grad, torch.full_like(m.bias.grad, 3.0))

# 2. Asymmetric empty EpochLogger must not deadlock across ranks
lgr = EpochLogger(output_dir={str(tmp_path / 'asymm')!r}, exp_name='asymm')
if proc_id() == 1:
    lgr.store(EpRet=5.0)
lgr.log_tabular('Epoch', 0)
lgr.log_tabular('EpRet', with_min_and_max=True)
lgr.dump_tabular()

# 3. MPI runs for PyTorch & TF2 PPO/TRPO
for name, algo in [
    ('ppo_pyt', ppo_pytorch),
    ('trpo_pyt', trpo_pytorch),
    ('ppo_tf2', ppo_tf2),
    ('trpo_tf2', trpo_tf2),
]:
    out = os.path.join({str(tmp_path)!r}, name)
    algo(
        env_fn=partial(gym.make, 'CartPole-v1'),
        ac_kwargs=dict(hidden_sizes=(16,)),
        steps_per_epoch=100,
        epochs=2,
        max_ep_len=40,
        logger_kwargs=dict(output_dir=out, exp_name=name),
    )
"""
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    proc = subprocess.run(
        ["mpirun", "--oversubscribe", "-np", "2", sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    for name in ("asymm", "ppo_pyt", "trpo_pyt", "ppo_tf2", "trpo_tf2"):
        df = pd.read_table(tmp_path / name / "progress.txt")
        assert len(df) >= 1
    assert pd.read_table(tmp_path / "asymm" / "progress.txt")["AverageEpRet"].iloc[0] == 5.0
