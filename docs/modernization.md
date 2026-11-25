# Spinning Up Modern: Technical Modernization Report

**Author:** Kiet Huynh  
**Date:** April 2026 (Updated November 2026)

## 1. Core Architectural Evolution
This report details the modernization of OpenAI's Spinning Up repository from legacy Python 3.6 / Gym / TensorFlow 1 code to modern Python 3.10–3.13, Gymnasium (`>=1.0.0`), PyTorch 2.x, and TensorFlow 2 / Keras 3.

### 1.1 Gymnasium Migration and Termination vs. Truncation Semantics
All 11+ algorithm implementations, examples, and exercises use the `gymnasium` 5-tuple step API (`obs, rew, terminated, truncated, info`) and 2-tuple reset API (`obs, info`). Crucially, true terminal failures (`terminated`) are separated from time-limit truncations (`truncated` or `ep_len == max_ep_len`):
- **On-Policy (`VPG`, `TRPO`, `PPO`):** Bootstrap with $V(s_T)$ whenever an episode is truncated or cut off by epoch boundaries; pass `last_val = 0.0` only on true `terminated`.
- **Off-Policy (`DDPG`, `TD3`, `SAC`):** Store `terminated` (not `terminated or truncated`) into `ReplayBuffer` so Bellman backups do not drop future value at artificial horizon cutoffs.
- **Reproducibility:** Seeds are offset per MPI rank (`seed + 10000 * proc_id()`) across both PyTorch and TF2 backends, and explicitly applied to `env.reset(seed=...)` and `env.action_space.seed(...)`.

## 2. Deep Learning Backends

### 2.1 PyTorch 2.10.0: Complete Algorithm Parity, MPI Staging, and `torch.compile`
- **Full PyTorch TRPO:** Added `spinup/algos/pytorch/trpo/` (`trpo_pytorch`) with exact Fisher-vector products via double `torch.autograd.grad` on forward KL divergence, conjugate gradient solver, MPI-synchronized backtracking line search, and value-function fitting.
- **Effective Graph Compilation (`--compile`):** Pure-tensor loss kernels (`_loss_pi_tensors`, `_loss_v_tensors`, `_loss_q_tensors`) are compiled directly with `torch.compile(...)` without wrapping unused outer container `forward()` methods.
- **Device-Safe Updates & MPI:** Target network Polyak averaging uses `target.copy_(polyak * target + (1 - polyak) * param)` under `torch.no_grad()`. `mpi_pytorch` stages tensors and gradients on contiguous CPU buffers before `MPI.COMM_WORLD` collectives and copies results back to `cuda` / `mps` / `cpu`.

### 2.2 TensorFlow 2.20.0 / Keras 3: Serialization, Graph Execution, and TRPO Trust Region
- **Keras 3 Custom Serialization:** Every subclassed `tf.keras.Model` is registered with `@tf.keras.utils.register_keras_serializable`, tracks custom weights via `self.add_weight(...)`, implements `get_config()` / `from_config()` using primitive space metadata only, provides `act(..., deterministic=...)`, and persists as `.keras` archives (`tf2_save/model.keras`).
- **TRPO Forward KL & MPI Line Search:** Fixed TF2 TRPO to compare candidate policies against a detached snapshot $\pi_{\theta_k}$ (`kl_from_old`), synchronized candidate KL and surrogate loss across MPI ranks before accepting backtracking steps, and fixed Keras 3 variable shape handling.
- **Graph-Compiled Updates:** Single-process off-policy algorithms (`ddpg_tf2`, `td3_tf2`, `sac_tf2`) execute `tf.GradientTape`, `apply_gradients`, and Polyak updates inside `@tf.function` training steps (with `optimizer.build(...)` called eagerly beforehand).

## 3. Hardware Acceleration: Native CUDA 12 / cuDNN 9 Dependency Alignment
`pyproject.toml` aligns `torch==2.10.0` and `tensorflow==2.20.0` on **CUDA 12 / cuDNN 9 (`libcudnn.so.9`)** across Python 3.10–3.13 (`x86_64` and `aarch64`), avoiding the legacy `torch==2.1.2` / `tensorflow==2.15.0` Python <= 3.11 and cuDNN 8 constraints.

`spinup/utils/device_utils.py` performs idempotent setup once per process (`setup_tf_gpu()`), pre-loading CUDA 12 / cuDNN 9 SONAMEs when present via `ctypes.RTLD_GLOBAL` and configuring XLA `libdevice.10.bc` / `ptxas` without creating directories in `$HOME` or racing across MPI ranks.

## 4. Logging, Plotting, and Packaging
- **MPI-Safe `EpochLogger`:** `mpi_statistics_scalar` and `EpochLogger.log_tabular` always participate across all ranks, returning `np.nan` on empty epochs so table headers never drift or cause late-epoch `AssertionError` / MPI deadlocks.
- **Modern Plotting & Checkpoint Testing:** `spinup/utils/plot.py` uses `seaborn.lineplot(..., errorbar='sd')`; `test_policy.py` forwards `--deterministic` for both PyTorch and TF2 `.keras` checkpoints.
- **PEP 621 Packaging:** Replaced legacy `py_modules=['spinup']` with `pyproject.toml` + `find_namespace_packages` (`spinup*`), optional extras (`[box2d]`, `[atari]`, `[genai]`, `[docs]`, `[all]`), and GitHub Actions CI.
