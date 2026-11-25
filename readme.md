# Spinning Up in Deep RL (Modernized)

**Status:** Active & Modernized (PyTorch 2.10.0, TensorFlow 2.20.0, Gymnasium)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/downloads/)
[![GPU Enabled](https://img.shields.io/badge/GPU-Accelerated-green.svg)](#hardware-acceleration)

Welcome to the **modernized** version of OpenAI's [Spinning Up in Deep RL](https://spinningup.openai.com)! This repository updates the ecosystem to modern PyTorch, TensorFlow 2 / Keras 3, and Gymnasium while preserving the educational clarity and standalone nature of the original implementations.

## What's New in the Modernized Version?

* **Gymnasium Integration:** Fully migrated from `gym` to `gymnasium` (`>=1.0.0`) with separated `terminated` vs `truncated` handling across all on-policy bootstrapping and off-policy replay buffers, standardized on `v5` MuJoCo tasks (e.g., `HalfCheetah-v5`).
* **TensorFlow 2.x / Keras 3 Migration:** Subclassed `tf.keras.Model` architectures with `@tf.keras.utils.register_keras_serializable`, eager + `@tf.function` `tf.GradientTape` training loops, `.keras` model persistence, and full `MpiAdamOptimizer` synchronization.
* **PyTorch 2.x Optimization & Complete TRPO:** Aligned on `torch==2.10.0` with pure-tensor `torch.compile()` loss graph optimization (`--compile`), device-safe `mpi_pytorch` gradient/parameter synchronization, and a full PyTorch TRPO implementation alongside VPG, PPO, DDPG, TD3, and SAC.
* **CUDA 12 / cuDNN 9 Dependency Alignment:** `torch==2.10.0` + `tensorflow==2.20.0` align natively on CUDA 12 and cuDNN 9 (`libcudnn.so.9`), with idempotent CUDA 12 / XLA initialization in `spinup.utils.device_utils`.

## Installation

### Prerequisites
- Python 3.10 through 3.13
- OpenMPI (for parallel execution: `openmpi-bin`, `libopenmpi-dev`)
- SWIG and Python development headers (optional, for `[box2d]` environments)
- NVIDIA Drivers (optional, for GPU support)

### Setup
```bash
# Create and activate conda or venv environment (Python 3.10–3.13)
conda create -n spinningup python=3.12
conda activate spinningup

# Install system dependencies (Ubuntu/Debian)
sudo apt update && sudo apt install -y openmpi-bin libopenmpi-dev swig python3-dev

# Clone the repository
git clone https://github.com/tkiethuynh/spinningup.git
cd spinningup

# Install core package (classic_control + mujoco included)
pip install -e .

# Optional extras: Box2D, Atari, Gemini analysis helper, or all
# pip install -e ".[box2d]"
# pip install -e ".[all]"
```

## Hardware Acceleration

Both backends automatically select available accelerators (`cuda` / `mps` / `cpu` for PyTorch; `/GPU:0` / `/device:CPU:0` for TensorFlow). See the [Modernization Report](docs/modernization.md) for technical details.

```bash
# Run PPO with PyTorch
python -m spinup.run ppo --env CartPole-v1

# Run PPO with TensorFlow 2
python -m spinup.run ppo_tf2 --env CartPole-v1

# Enable PyTorch loss-graph compilation
python -m spinup.run ppo --env HalfCheetah-v5 --compile
```

## Documentation

- **[Modernization Report](docs/modernization.md)**: Architectural notes, dependency alignment, and verification details (`docs/modernization.tex` for LaTeX source).
- **[Spinning Up Site](https://spinningup.openai.com)**: Original RL theory and educational content.

## Included Algorithms

All six core algorithms are implemented in both **PyTorch** and **TensorFlow 2**:
- Vanilla Policy Gradient (`vpg_pytorch`, `vpg_tf2`)
- Trust Region Policy Optimization (`trpo_pytorch`, `trpo_tf2`)
- Proximal Policy Optimization (`ppo_pytorch`, `ppo_tf2`)
- Deep Deterministic Policy Gradient (`ddpg_pytorch`, `ddpg_tf2`)
- Twin Delayed DDPG (`td3_pytorch`, `td3_tf2`)
- Soft Actor-Critic (`sac_pytorch`, `sac_tf2`)

## Testing

```bash
pytest -v test/
```

## Citing

If you use this modernized version in your research, please cite both the original work and this modernized fork:

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
