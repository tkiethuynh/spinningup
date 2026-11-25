import importlib

# Loggers and version are lightweight and always available without importing DL backends
from spinup.utils.logx import Logger, EpochLogger
from spinup.version import __version__

_PYTORCH_ALGOS = {
    "vpg_pytorch": ("spinup.algos.pytorch.vpg.vpg", "vpg"),
    "trpo_pytorch": ("spinup.algos.pytorch.trpo.trpo", "trpo"),
    "ppo_pytorch": ("spinup.algos.pytorch.ppo.ppo", "ppo"),
    "ddpg_pytorch": ("spinup.algos.pytorch.ddpg.ddpg", "ddpg"),
    "td3_pytorch": ("spinup.algos.pytorch.td3.td3", "td3"),
    "sac_pytorch": ("spinup.algos.pytorch.sac.sac", "sac"),
}

_TF2_ALGOS = {
    "vpg_tf2": ("spinup.algos.tf2.vpg.vpg", "vpg"),
    "trpo_tf2": ("spinup.algos.tf2.trpo.trpo", "trpo"),
    "ppo_tf2": ("spinup.algos.tf2.ppo.ppo", "ppo"),
    "ddpg_tf2": ("spinup.algos.tf2.ddpg.ddpg", "ddpg"),
    "td3_tf2": ("spinup.algos.tf2.td3.td3", "td3"),
    "sac_tf2": ("spinup.algos.tf2.sac.sac", "sac"),
}

__all__ = [*_PYTORCH_ALGOS.keys(), *_TF2_ALGOS.keys(), "Logger", "EpochLogger", "__version__"]


def __getattr__(name: str):
    if name in _PYTORCH_ALGOS:
        module_path, attr = _PYTORCH_ALGOS[name]
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise ImportError(
                    f"PyTorch is required for '{name}'. Install with: pip install -e '.[pytorch]'"
                ) from exc
            raise
        obj = getattr(mod, attr)
        globals()[name] = obj
        return obj

    if name in _TF2_ALGOS:
        from spinup.utils.device_utils import setup_tf_gpu

        setup_tf_gpu()
        module_path, attr = _TF2_ALGOS[name]
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            if exc.name == "tensorflow":
                raise ImportError(
                    f"TensorFlow is required for '{name}'. Install with: pip install -e '.[tf2]'"
                ) from exc
            raise
        obj = getattr(mod, attr)
        globals()[name] = obj
        return obj

    raise AttributeError(f"module 'spinup' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
