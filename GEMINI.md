# Gemini CLI foundational instructions for Spinning Up Modern

This project is a modernized version of OpenAI's Spinning Up in Deep RL. The codebase has been updated to use latest stable versions of all core libraries while maintaining the educational clarity of the original.

## Modernization Standards

### Environment & API
- **Gymnasium Integration:** The project is fully migrated to `gymnasium`. All environment interactions follow the modern API (`obs, reward, terminated, truncated, info`).
- **Python 3.10+:** The codebase targets modern Python features and standards.
- **Type Hinting:** All algorithm functions and core classes include comprehensive Python type hints.

### Deep Learning Backends
- **PyTorch 2.x:**
    - Fully modernized algorithm implementations.
    - Support for `torch.compile` via the `--compile` flag.
    - Automatic GPU/CUDA acceleration.
- **TensorFlow 2.x:**
    - Complete migration from TF1 to TF2.
    - Uses Keras models, Eager Execution, and `tf.GradientTape`.
    - Fully GPU-enabled using a dynamic library resolution strategy for NVIDIA pip packages.
- **Legacy TF1:** Preserved for historical reference but isolated to prevent system-wide library conflicts.

### Device & Resource Management
- **Automatic Device Selection:** Uses `spinup.utils.device_utils` for framework-agnostic GPU/CPU detection.
- **MPI Support:** Maintained and modernized compatibility with MPI for parallel execution across both frameworks.

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
