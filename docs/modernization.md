# Spinning Up Modern: Technical Modernization Report

**Author:** Kiet Huynh  
**Date:** April 2026 (Updated November 2026)

## 1. Core Architectural Evolution
This report details the modernization of OpenAI's Spinning Up repository from legacy Python 3.6 / Gym / TensorFlow 1 code to modern Python 3.10–3.13, Gymnasium (`>=1.0.0`), PyTorch 2.x, and TensorFlow 2 / Keras 3 while preserving Spinning Up's textbook readability.

### 1.1 Gymnasium Migration and Termination vs. Truncation Semantics
All 12 algorithm implementations, `pg_math` tutorials, and exercises use the `gymnasium` 5-tuple step API (`obs, rew, terminated, truncated, info`) and 2-tuple reset API (`obs, info`). Crucially, true terminal failures (`terminated`) are separated from time-limit truncations (`truncated` or `ep_len == max_ep_len`):
- **On-Policy (`VPG`, `TRPO`, `PPO`):** Bootstrap with $V(s_T)$ whenever an episode is truncated or cut off by epoch boundaries; pass `last_val = 0.0` only on true `terminated`.
- **Off-Policy (`DDPG`, `TD3`, `SAC`):** Store `terminated` (not `terminated or truncated`) into `ReplayBuffer` so Bellman backups do not drop future value at artificial horizon cutoffs.
- **Reproducibility:** Seeds are offset per MPI rank (`seed + 10000 * proc_id()`) across both PyTorch and TF2 backends (via `tf.keras.utils.set_random_seed`), and explicitly applied to `env.reset(seed=...)` and `env.action_space.seed(...)`.

## 2. Deep Learning Backends

### 2.1 PyTorch 2.10.0: TRPO Parity, Clean Loss Functions, and Device Control
- **Full PyTorch TRPO:** Added `spinup/algos/pytorch/trpo/` (`trpo_pytorch`) with exact Fisher-vector products via double `torch.autograd.grad` on forward KL divergence, conjugate gradient solver, MPI-synchronized backtracking line search, and value-function fitting.
- **Readable Objective Kernels & Safe Updates:** Policy and Q losses keep Spinning Up's single-function textbook structure (`pi, logp = ac.pi(obs, act)`). Target network Polyak averaging uses `target.copy_(polyak * target + (1 - polyak) * param)` under `torch.no_grad()`.
- **CPU Default + Device Flexibility:** Algorithms default to `device='cpu'` (optimal for single-env shallow MLP rollouts) while supporting `device='auto'` / `'cuda'` / `'mps'` via `get_torch_device(device)`. `mpi_pytorch` stages tensors and gradients on contiguous CPU buffers before MPI collectives and copies updates back to target devices.

### 2.2 TensorFlow 2.20.0 / Keras 3: Slim Serialization and Graph Execution
- **Shared Keras 3 Serialization (`_keras_utils.py`):** Outer `MLPActorCritic` classes inherit `SerializableActorCritic`, reducing per-algorithm `core.py` boilerplate by keeping inner actor/critic layers clean while supporting `.keras` archive saving (`tf2_save/model.keras`) via `@tf.keras.utils.register_keras_serializable` and `add_weight`.
- **TRPO Forward KL & MPI Line Search:** TF2 TRPO compares candidate policies against a detached snapshot $\pi_{\theta_k}$ (`kl_from_old`), synchronizes candidate KL and surrogate loss across MPI ranks before accepting backtracking steps, and aligns variable shapes for Keras 3.
- **Graph-Compiled Updates & `pg_math` Symmetry:** Single-process off-policy algorithms (`ddpg_tf2`, `td3_tf2`, `sac_tf2`) execute `tf.GradientTape`, optimizer updates, and Polyak averaging inside `@tf.function` steps (with `optimizer.build(...)` called eagerly beforehand). Standalone `1_simple_pg.py` and `2_rtg_pg.py` tutorials live in both `spinup/examples/pytorch/pg_math/` and `spinup/examples/tf2/pg_math/`.

## 3. Lazy Backends & CUDA 12 / cuDNN 9 Alignment
- **Lazy Package Exports:** `spinup/__init__.py` uses PEP 562 `__getattr__` so `import spinup` loads neither PyTorch nor TensorFlow eagerly. `pyproject.toml` makes `[pytorch]` (`torch==2.10.0`) and `[tf2]` (`tensorflow==2.20.0`) optional extras (`[all]` installs both).
- **Native CUDA 12 / cuDNN 9 Interoperability:** When both backends are installed together (`[all]`), `torch==2.10.0` and `tensorflow==2.20.0` align natively on **CUDA 12 / cuDNN 9 (`libcudnn.so.9`)** across Python 3.10–3.13 (`x86_64` and `aarch64`). `spinup/utils/device_utils.py` runs `setup_tf_gpu()` once per process on TF2 initialization without mutating `$HOME`.

## 4. Logging, Plotting, and Packaging
- **MPI-Safe `EpochLogger`:** `mpi_statistics_scalar` and `EpochLogger.log_tabular` always participate across all ranks, returning `np.nan` on empty epochs so table headers never drift or cause late-epoch `AssertionError` / MPI deadlocks.
- **Modern Plotting & Checkpoint Testing:** `spinup/utils/plot.py` uses `seaborn.lineplot(..., errorbar='sd')`; `test_policy.py` forwards `--deterministic` for both PyTorch and TF2 `.keras` checkpoints.
- **PEP 621 Packaging & CI:** Replaced legacy `py_modules=['spinup']` with `pyproject.toml` + `find_namespace_packages` (`spinup*`), optional extras (`[pytorch]`, `[tf2]`, `[box2d]`, `[atari]`, `[docs]`, `[all]`), Sphinx `autodoc` verification, and GitHub Actions CI.
