# Gemini CLI foundational instructions for Spinning Up Modern

This project is a modernized version of OpenAI's Spinning Up in Deep RL. The codebase targets modern PyTorch, TensorFlow 2 / Keras 3, and Gymnasium while maintaining the educational clarity of the original.

### Dependency Management & Hardware Acceleration
- **Dependency Alignment**: `pyproject.toml` pins `torch==2.10.0` and `tensorflow==2.20.0` so both frameworks align natively on CUDA 12 / cuDNN 9 (`libcudnn.so.9`) across Python 3.10–3.13 without ABI clashes. Keep `device_utils.py` idempotent and free of `$HOME` side effects.

### Framework-Specific Guidelines

#### PyTorch 2.x
- **In-Place Updates**: Use `torch.no_grad()` and `target_param.copy_(polyak * target_param + (1 - polyak) * param)` for target network updates instead of `.data` attribute manipulation.
- **Device Placement**: Always move models to `get_torch_device()`, move input tensors to device before inference, and stage MPI buffers on CPU before `Bcast`/`Allreduce`.
- **Graph Optimization**: Compile pure-tensor loss functions (not outer `MLPActorCritic` containers without `forward`) when `--compile` is enabled.

#### TensorFlow 2.x / Keras 3
- **Keras Serialization**: Register every subclassed `tf.keras.Model` with `@tf.keras.utils.register_keras_serializable`, accept `**kwargs`, track weights via `self.add_weight(...)`, implement `get_config()` / `from_config()` using primitive space metadata only, and save models with `.keras` extensions.
- **Eager & Graph Logic**: Use `tf.GradientTape` inside `@tf.function` training steps (with `optimizer.build(...)` called eagerly beforehand) and `with tf.device(device):` contexts.
- **Mathematical Precision**: In TRPO, always compute forward KL $D_{\mathrm{KL}}(\pi_{\theta_k} \,\|\, \pi_\theta)$ against a detached snapshot of the pre-update policy.

### Environment & Logging Standards
- **Gymnasium Semantics**: Never collapse `terminated` and `truncated` into a single `done` for bootstrapping or replay storage. Only true `terminated` zeroes future value bootstrap.
- **Logger & MPI Safety**: `EpochLogger` and `mpi_statistics_scalar` must always participate across all MPI ranks even on empty epochs (`np.nan` placeholder) so headers stay consistent and collectives never deadlock.
- **Verification**: Run `pytest -v test/` after edits and verify `python -m spinup.run ppo --env CartPole-v1`.
