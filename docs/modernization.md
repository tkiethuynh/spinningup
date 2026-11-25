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
- **Native CUDA 12 / cuDNN 9 Interoperability & Multi-GPU MPI Sharding:** When both backends are installed together (`[all]`), `torch==2.10.0` and `tensorflow==2.20.0` align natively on **CUDA 12 / cuDNN 9 (`libcudnn.so.9`)** across Python 3.10–3.13 (`x86_64` and `aarch64`). `spinup/utils/device_utils.py` runs `setup_tf_gpu()` once per process on TF2 initialization without mutating `$HOME`, supports explicit device indices (`cuda:N`, `gpu:N`, `/GPU:N`), and automatically maps MPI workers across visible GPUs (`proc_id() % n_gpus`).

## 4. Logging, Plotting, and Packaging
- **MPI-Safe `EpochLogger`:** `mpi_statistics_scalar` and `EpochLogger.log_tabular` always participate across all ranks, returning `np.nan` on empty epochs so table headers never drift or cause late-epoch `AssertionError` / MPI deadlocks. `Logger._tf2_simple_save` and `restore_tf2_model` also isolate Keras 3 / NumPy 2 array-copy deprecation noise during `.keras` serialization.
- **Modern Plotting, Atari & Checkpoint Testing:** `spinup/utils/plot.py` uses `seaborn.lineplot(..., errorbar='sd')`; `test_policy.py` forwards `--deterministic` for both PyTorch and TF2 `.keras` checkpoints; and `spinup/run.py` / `spinup/utils/run_utils.py` auto-register bundled `ale-py` Atari environments (`ALE/*-v5` and `ALE/*-ram-v5`) and flatten multi-dimensional `Box` observations when using MLP policies.
- **PEP 621 Packaging, Zero-Warning Docs & CI:** Replaced legacy `py_modules=['spinup']` with `pyproject.toml` + `find_namespace_packages` (`spinup*`), optional extras (`[pytorch]`, `[tf2]`, `[box2d]`, `[atari]`, `[docs]`, `[all]`), strict zero-warning Sphinx builds (`imgmath` SVG when TeX Live is present, `mathjax` fallback otherwise), GitHub Pages deployment, and GitHub Actions CI (`test (3.10)`, `test (3.12)`, `docs`).

## 5. Empirical MuJoCo `v5` Verification
To verify learning dynamics on open-source Gymnasium MuJoCo `v5` tasks beyond single-epoch smoke checks (with `seed=0`):
- **On-Policy PPO (100,000 steps, 4 MPI ranks, CPU, `steps_per_epoch=4000`, `epochs=25`):** `ppo_pytorch` and `ppo_tf2` were evaluated across 25 epochs on `HalfCheetah-v5` and `Hopper-v5`.
- **Off-Policy SAC (20,000 steps, NVIDIA RTX 3070, `steps_per_epoch=2000`, `epochs=10`):** `sac_pytorch` (`device='cuda'`) and `sac_tf2` (`device='gpu'`) were evaluated with `num_test_episodes=5` deterministic evaluations per epoch. On `HalfCheetah-v5`, moving `sac_pytorch` from ARM CPU (`405.46 s`) to RTX 3070 (`226.05 s`) cut runtime by ~44%.
- **Fast Convergence Regression Gate (`test/test_examples_and_exercises.py`):** Automated CI asserts multi-fold learning improvement on `InvertedPendulum-v5` for `ppo_pytorch` (8 epochs x 500 steps) and `sac_pytorch` (4 epochs x 400 steps).

| Algorithm | Environment | Hardware / Mode | Env Steps | Initial Return | Peak Return | Final Return | Wall Time |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `ppo_pytorch` | `HalfCheetah-v5` | CPU (`4` MPI ranks) | 100,000 | `-288.8` | `-117.0` | `-151.6` | 55.64 s |
| `ppo_pytorch` | `Hopper-v5` | CPU (`4` MPI ranks) | 100,000 | `8.7` | `231.6` | `231.6` | 67.91 s |
| `ppo_tf2` | `HalfCheetah-v5` | CPU (`4` MPI ranks) | 100,000 | `-481.1` | `-301.1` | `-341.4` | 445.54 s |
| `ppo_tf2` | `Hopper-v5` | CPU (`4` MPI ranks) | 100,000 | `11.3` | `302.3` | `302.3` | 526.31 s |
| `sac_pytorch` | `HalfCheetah-v5` | RTX 3070 (`cuda`) | 20,000 | `-26.7` | `950.6` | `766.2` | 226.05 s |
| `sac_pytorch` | `Hopper-v5` | RTX 3070 (`cuda`) | 20,000 | `97.3` | `805.7` | `471.9` | 195.15 s |
| `sac_tf2` | `HalfCheetah-v5` | RTX 3070 (`gpu`) | 20,000 | `-30.3` | `698.4` | `698.4` | 428.21 s |
| `sac_tf2` | `Hopper-v5` | RTX 3070 (`gpu`) | 20,000 | `95.1` | `781.9` | `686.3` | 261.06 s |
