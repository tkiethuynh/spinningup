# Spinning Up in Deep RL (Modernized)

**Status:** Active & Modernized (PyTorch 2.5+, TensorFlow 2.15, Gymnasium)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GPU Enabled](https://img.shields.io/badge/GPU-Accelerated-green.svg)](#hardware-acceleration)

Welcome to the **modernized** version of OpenAI's [Spinning Up in Deep RL](https://spinningup.openai.com)! This repository has been updated to support the latest software ecosystems while preserving the educational clarity and standalone nature of the original implementations.

## What's New in the Modernized Version?

This fork transforms the legacy Spinning Up codebase into a production-ready research environment:

*   **Gymnasium Integration:** Fully migrated from `gym` to the latest `gymnasium` API. Standardized on `v5` MuJoCo environments (e.g., `HalfCheetah-v5`).
*   **TensorFlow 2.x Migration:** Complete rewrite of the TensorFlow backend. Gone are sessions and placeholders; the new implementation uses **Keras Models**, **Eager Execution**, and `tf.GradientTape`.
*   **PyTorch 2.x Optimization:** Updated to PyTorch 2.5+. Includes support for **`torch.compile()`** for superior graph-mode performance.
*   **Hardware Acceleration:** Native support for GPU/CUDA across both frameworks.
*   **Dynamic Library Resolution:** A custom initialization system that automatically resolves version mismatches in pip-installed NVIDIA libraries (e.g., matching cuDNN 9 to TF 2.15 expectations).
*   **Type Hinting & Modern Python:** Comprehensive Python type hints added to all core algorithms and utilities for enhanced developer experience.

## Installation

### Prerequisites
- Python 3.10 or greater
- OpenMPI (for parallel execution)
- SWIG (required for Box2D environments: `sudo apt install swig`)
- NVIDIA Drivers (for GPU support)

### Setup
```bash
# Create and activate conda environment
conda create -n spinningup python=3.10
conda activate spinningup

# Install system dependencies (Ubuntu/Debian)
sudo apt update && sudo apt install -y swig libopenmpi-dev

# Clone the repository
git clone https://github.com/tkiethuynh/spinningup.git
cd spinningup

# Install core package
pip install -e .
pip install nvidia-tensorrt
```

## Hardware Acceleration

The system automatically detects your GPU and configures the environment. For detailed technical information on how we handle library version mapping, see the [Modernization Report](docs/modernization.pdf).

### Running on GPU
Backends automatically detect and use the most capable device.
```bash
# Run PPO with PyTorch on GPU
python -m spinup.run ppo --env CartPole-v1

# Run PPO with TensorFlow 2 on GPU
python -m spinup.run ppo_tf2 --env CartPole-v1

# Enable PyTorch graph compilation for speed
python -m spinup.run ppo --env HalfCheetah-v5 --compile
```

## Documentation

- **[Modernization Report](docs/modernization.pdf)**: Comprehensive technical details on the architectural changes and GPU resolution strategies.
- **[Spinning Up Site](https://spinningup.openai.com)**: The original educational content (theory and background).

## Included Algorithms

All algorithms are implemented in both **PyTorch** and **TensorFlow 2**:
- Vanilla Policy Gradient (VPG)
- Trust Region Policy Optimization (TRPO)
- Proximal Policy Optimization (PPO)
- Deep Deterministic Policy Gradient (DDPG)
- Twin Delayed DDPG (TD3)
- Soft Actor-Critic (SAC)

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
