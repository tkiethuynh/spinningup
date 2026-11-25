# Spinning Up in Deep RL (Modernized)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/tkiethuynh/spinningup/actions/workflows/ci.yml/badge.svg)](https://github.com/tkiethuynh/spinningup/actions/workflows/ci.yml)

Welcome to the **modernized** edition of OpenAI's [Spinning Up in Deep RL](https://spinningup.openai.com)! This repository brings the curriculum to modern **Python 3.10–3.13**, **Gymnasium (`>=1.0.0`)** (including open-source MuJoCo `v5`), **PyTorch 2.x**, and **TensorFlow 2 / Keras 3** while keeping what made the original special: short, standalone, heavily annotated implementations where every equation maps cleanly to readable Python.

## Learn Deep RL

Whether you are just starting out or bridging theory to code, Spinning Up provides:

- A structured **[Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)** (terminology, math, policy optimization, Q-learning).
- Joshua Achiam's essay **[Spinning Up as a Deep RL Researcher](https://spinningup.openai.com/en/latest/spinningup/spinningup.html)** on how to learn and debug RL algorithms.
- A curated list of **[Key Papers](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)** organized by topic.
- Minimal **~130-line starter scripts** demonstrating policy gradients from scratch in both frameworks:
  - [`spinup/examples/pytorch/pg_math/`](spinup/examples/pytorch/pg_math/) (`1_simple_pg.py`, `2_rtg_pg.py`)
  - [`spinup/examples/tf2/pg_math/`](spinup/examples/tf2/pg_math/) (`1_simple_pg.py`, `2_rtg_pg.py`)
- Annotated, standalone **RL algorithm implementations** in **PyTorch** and **TensorFlow 2** (`vpg`, `trpo`, `ppo`, `ddpg`, `td3`, `sac`).
- Hands-on **warm-up exercises** in [`spinup/exercises/pytorch/`](spinup/exercises/pytorch/).

## Quickstart & Installation

### Prerequisites
- Python 3.10 through 3.13
- OpenMPI (for parallel CPU execution: `openmpi-bin` and `libopenmpi-dev` on Ubuntu/Debian; `openmpi` via Homebrew on macOS)
- SWIG (`swig`) and Python dev headers (`python3-dev`) only if installing `[box2d]`

### Setup

```bash
conda create -n spinningup python=3.12
conda activate spinningup

# System dependencies (Ubuntu/Debian)
sudo apt update && sudo apt install -y openmpi-bin libopenmpi-dev

git clone https://github.com/tkiethuynh/spinningup.git
cd spinningup

# Install Spinning Up with your chosen backend (PyTorch recommended)
pip install -e ".[pytorch]"

# Or TensorFlow 2 / Keras 3 only:
# pip install -e ".[tf2]"

# Or both backends (aligned on CUDA 12 / cuDNN 9) + Box2D + Atari:
# pip install -e ".[all]"
```

> **Tip (lightweight CPU laptop install):** To avoid downloading CUDA wheels when learning on a CPU-only machine, install CPU PyTorch first, then the core package:
> ```bash
> pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
> pip install -e .
> ```

### Running Your First Experiment

```bash
# Train PPO on CartPole-v1 (uses PyTorch backend by default)
python -m spinup.run ppo --env CartPole-v1 --exp_name hello_ppo

# Compare the simplest policy gradient script (~130 lines)
python spinup/examples/pytorch/pg_math/1_simple_pg.py
python spinup/examples/pytorch/pg_math/2_rtg_pg.py

# Watch the trained agent or plot progress
python -m spinup.run test_policy data/hello_ppo/hello_ppo_s0
python -m spinup.run plot data/hello_ppo
```

### CPU vs. GPU (`--device`)

Algorithms default to **`--device cpu`**, which is typically **faster** for single-environment RL loops with small `[64, 64]` or `[256, 256]` MLPs because every `env.step()` executes one transition at a time on CPU (avoiding per-step host-to-GPU transfer latency). Pass `--device auto` or `--device cuda` / `--device gpu` when using larger networks, custom pixel critics, or vectorized experiments:

```bash
python -m spinup.run ppo --env HalfCheetah-v5 --device auto
python -m spinup.run sac_tf2 --env HalfCheetah-v5 --device gpu
```

## Included Algorithms

All six core algorithms are implemented symmetrically in both **PyTorch** (`spinup/algos/pytorch/`) and **TensorFlow 2** (`spinup/algos/tf2/`) with explicit `terminated` vs. `truncated` bootstrapping, full parameter docstrings, and MPI support:

- Vanilla Policy Gradient (`vpg_pytorch`, `vpg_tf2`)
- Trust Region Policy Optimization (`trpo_pytorch`, `trpo_tf2`)
- Proximal Policy Optimization (`ppo_pytorch`, `ppo_tf2`)
- Deep Deterministic Policy Gradient (`ddpg_pytorch`, `ddpg_tf2`)
- Twin Delayed DDPG (`td3_pytorch`, `td3_tf2`)
- Soft Actor-Critic (`sac_pytorch`, `sac_tf2`)

## Testing & Engineering Details

```bash
pytest -v test/
```

See the **[Modernization Report](docs/modernization.md)** (`docs/modernization.tex` for LaTeX source) for architectural details on Gymnasium migration, Keras 3 `.keras` serialization, and CUDA 12 / cuDNN 9 alignment.

## Citing

If you use this codebase in your research, please cite both the original Spinning Up work and this modernized repository:

```bibtex
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}

@misc{SpinningUpModern2026,
    author = {Kiet Huynh},
    title = {{Spinning Up in Deep RL (Modernized)}},
    year = {2026},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/tkiethuynh/spinningup}}
}
```
