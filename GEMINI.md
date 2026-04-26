# Gemini CLI foundational instructions for Spinning Up Modern

This project is a modernized version of OpenAI's Spinning Up in Deep RL. The codebase has been updated to use latest stable versions of all core libraries while maintaining the educational clarity of the original.

### Framework-Specific Guidelines

#### PyTorch 2.x
- **In-Place Updates**: Use `torch.no_grad()` and `target_param.copy_(polyak * target_param + (1 - polyak) * param)` for target network updates instead of deprecated `.data` attribute manipulation.
- **Device Placement**: Always move models to the device returned by `get_torch_device()` and ensure input tensors are moved before inference, especially in testing utilities.
- **Graph Optimization**: Support `torch.compile()` via a standard `--compile` flag in algorithm entry points.

#### TensorFlow 2.x
- **Keras Serialization**: Always implement `get_config()` in subclassed `tf.keras.Model` classes. Ensure that complex objects like Gymnasium spaces are not serialized directly; store only necessary primitive parameters.
- **Eager Logic**: Use `tf.GradientTape` for all training loops and wrap operations in `with tf.device(device):` contexts.
- **Mathematical Precision**: Use explicit `entropy()` methods on actor models for exact calculations.

### Logging & Utilities
- **Robustness**: The `EpochLogger` is enhanced to handle empty data batches during epoch transitions without crashing.
- **Model Restoration**: Use `restore_tf2_model` (Keras-native) for TF2 models.



## Project Standards

### Code Style & Architecture
- **Standalone Implementations:** Algorithms (PPO, DDPG, SAC, etc.) remain standalone within their backend directories (`spinup/algos/pytorch` and `spinup/algos/tf2`).
- **Educational Clarity:** Variable names and documentation follow RL theory standards.
- **Logging:** Uses the enhanced `spinup.utils.logx.EpochLogger` for all training loops, with support for modern serialization formats (PT and SavedModel).

### Tooling & Commands
- **Testing:** Use `pytest` for all verification.
- **Modernization Documentation:** Technical details of the upgrade are available in `docs/modernization.tex`.

### Development Workflow
- **Verification:** Always verify changes by running the algorithm on a simple environment using the provided run utilities:
  `python -m spinup.run ppo --env CartPole-v1`
- **Documentation:** Maintain the LaTeX report and standard `.rst` files when modifying algorithm logic.
