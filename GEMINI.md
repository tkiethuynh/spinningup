# Gemini CLI foundational instructions for Spinning Up Modern

This project is a modernized version of OpenAI's Spinning Up in Deep RL. The codebase targets modern PyTorch, TensorFlow 2 / Keras 3, and Gymnasium while preserving the educational clarity and standalone readability of the original.

### Packaging & Lazy Backends
- **Optional Backends**: Base `spinup` depends only on the core RL/scientific stack (`gymnasium[classic-control,mujoco]`, `mpi4py`, `numpy`, `scipy`, etc.). `torch==2.10.0` (`[pytorch]`) and `tensorflow==2.20.0` (`[tf2]`) are optional extras that align natively on CUDA 12 / cuDNN 9 (`libcudnn.so.9`) when installed together (`[all]`).
- **Lazy Top-Level Exports**: `spinup/__init__.py` resolves `*_pytorch` and `*_tf2` via PEP 562 `__getattr__`, so importing `spinup`, `EpochLogger`, or utilities never eagerly loads heavyweight frameworks. Keep `device_utils.py` lazy and free of `$HOME` side effects.

### Framework-Specific Guidelines

#### PyTorch 2.x
- **Textbook Loss Clarity**: Keep `compute_loss_pi`, `compute_loss_v`, and `compute_loss_q` simple and readable (`pi, logp = ac.pi(obs, act)`), without graph-break workarounds or `torch.compile` wrappers that obscure policy-gradient math.
- **In-Place Target Updates**: Use `torch.no_grad()` and `target_param.copy_(polyak * target_param + (1 - polyak) * param)` for target network updates instead of `.data` attribute manipulation.
- **Device Selection (`device='cpu'` Default)**: Default `device='cpu'` (usually fastest for single-env shallow MLP rollouts), allow `'cuda'`, `'mps'`, or `'auto'` via `get_torch_device(device)`, and stage MPI buffers on CPU before `Bcast`/`Allreduce`.

#### TensorFlow 2.x / Keras 3
- **Keras Serialization**: Inherit `SerializableActorCritic` from `spinup/algos/tf2/_keras_utils.py` and decorate each `MLPActorCritic` with `@tf.keras.utils.register_keras_serializable` (keeping inner actor/critic submodules clean), track weights via `self.add_weight(...)`, and save models as `.keras` archives.
- **Eager & Graph Logic**: Call `setup_tf_gpu()` before importing `tensorflow`, build optimizers eagerly (`optimizer.build(...)`) before `@tf.function` steps, use `tf.keras.utils.set_random_seed(seed)` for Keras 3 reproducibility, and pass `device='cpu'` (or `'auto'`/`'gpu'`) into `get_tf_device(device)`.
- **Mathematical Precision**: In TRPO, always compute forward KL $D_{\mathrm{KL}}(\pi_{\theta_k} \,\|\, \pi_\theta)$ against a detached snapshot of the pre-update policy.

### Environment, Documentation & Logging Standards
- **Gymnasium Semantics**: Never collapse `terminated` and `truncated` into a single `done` for bootstrapping or replay storage. Only true `terminated` zeroes future value bootstrap.
- **Teaching Parity**: Keep full RST docstrings (`Args:` + `Symbol | Shape | Description` tables) and inline `#` explanations on all 12 algorithms, plus parallel `pg_math` examples in PyTorch and TF2.
- **Logger & MPI Safety**: `EpochLogger` and `mpi_statistics_scalar` must always participate across all MPI ranks even on empty epochs (`np.nan` placeholder) so headers stay consistent and collectives never deadlock.
- **Verification**: Run `pytest -v test/` after edits and verify `python -m spinup.run ppo --env CartPole-v1`.
