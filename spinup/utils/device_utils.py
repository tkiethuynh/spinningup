import ctypes
import logging
import os
import site
import sys
import tempfile

import torch

_GPU_SETUP_DONE = False


def _prepend_env_path(var_name: str, entries: list[str]) -> None:
    if not entries:
        return
    current = os.environ.get(var_name, "")
    existing = [p for p in current.split(":") if p]
    merged = [e for e in entries if e not in existing] + existing
    os.environ[var_name] = ":".join(merged)


def setup_tf_gpu():
    """
    Configures the environment for TensorFlow and PyTorch GPU interoperability.
    Idempotent across multiple calls. On Linux, sets TF log/memory defaults and
    pre-loads CUDA 12 / cuDNN 9 libraries via ctypes RTLD_GLOBAL when present
    without mutating $HOME.
    """
    global _GPU_SETUP_DONE
    if _GPU_SETUP_DONE:
        return
    _GPU_SETUP_DONE = True

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    if not sys.platform.startswith("linux"):
        return

    active_site_packages = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if isinstance(user_site, str) and user_site not in active_site_packages:
        active_site_packages.append(user_site)

    nvidia_paths = [
        os.path.join(sp, "nvidia")
        for sp in active_site_packages
        if os.path.isdir(os.path.join(sp, "nvidia"))
    ]

    if nvidia_paths:
        lib_dirs = []
        for nvidia_path in nvidia_paths:
            for root, dirs, _ in os.walk(nvidia_path):
                if "lib" in dirs:
                    lib_dir = os.path.join(root, "lib")
                    if lib_dir not in lib_dirs:
                        lib_dirs.append(lib_dir)

        if lib_dirs:
            _prepend_env_path("LD_LIBRARY_PATH", lib_dirs)

            # CUDA 12 / cuDNN 9 SONAMEs aligned with PyTorch 2.10 + TF 2.20
            core_libs = [
                "libcudart.so.12",
                "libnvJitLink.so.12",
                "libcublasLt.so.12",
                "libcublas.so.12",
                "libcudnn.so.9",
                "libcufft.so.11",
                "libcurand.so.10",
                "libcusolver.so.11",
                "libcusparse.so.12",
            ]
            for lib_name in core_libs:
                for lib_dir in lib_dirs:
                    full_path = os.path.join(lib_dir, lib_name)
                    if os.path.exists(full_path):
                        try:
                            ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                            break
                        except OSError:
                            pass

    # Configure XLA libdevice cleanly (prefer direct site-packages path, fallback to /tmp)
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_cuda_data_dir" not in xla_flags:
        cuda_data_dir = None
        for sp in active_site_packages:
            nvcc_dir = os.path.join(sp, "nvidia", "cuda_nvcc")
            if os.path.exists(os.path.join(nvcc_dir, "nvvm", "libdevice", "libdevice.10.bc")):
                cuda_data_dir = nvcc_dir
                break

        if cuda_data_dir is None:
            for sp in active_site_packages:
                for triton_cand in (
                    os.path.join(sp, "triton", "backends", "nvidia", "lib", "libdevice.10.bc"),
                    os.path.join(sp, "triton", "third_party", "cuda", "lib", "libdevice.10.bc"),
                ):
                    if os.path.exists(triton_cand):
                        uid = os.getuid() if hasattr(os, "getuid") else 0
                        xla_dir = os.path.join(tempfile.gettempdir(), f"spinup_xla_{uid}")
                        nvvm_dir = os.path.join(xla_dir, "nvvm", "libdevice")
                        os.makedirs(nvvm_dir, exist_ok=True)
                        libdevice_dest = os.path.join(nvvm_dir, "libdevice.10.bc")
                        if not os.path.exists(libdevice_dest):
                            tmp_link = f"{libdevice_dest}.{os.getpid()}"
                            try:
                                if os.path.lexists(tmp_link):
                                    os.remove(tmp_link)
                                os.symlink(triton_cand, tmp_link)
                                os.replace(tmp_link, libdevice_dest)
                            except OSError:
                                pass
                        if os.path.exists(libdevice_dest):
                            cuda_data_dir = xla_dir
                        break
                if cuda_data_dir is not None:
                    break

        if cuda_data_dir is not None:
            os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_data_dir} {xla_flags}".strip()

    # Ensure ptxas is in PATH for XLA JIT
    ptxas_dirs = []
    for sp in active_site_packages:
        for cand_dir in (
            os.path.join(sp, "nvidia", "cuda_nvcc", "bin"),
            os.path.join(sp, "triton", "backends", "nvidia", "bin"),
            os.path.join(sp, "triton", "third_party", "cuda", "bin"),
        ):
            if os.path.exists(os.path.join(cand_dir, "ptxas")):
                ptxas_dirs.append(cand_dir)
                break
        if ptxas_dirs:
            break
    _prepend_env_path("PATH", ptxas_dirs)


# Call setup BEFORE importing tensorflow
setup_tf_gpu()

try:
    import tensorflow as tf
    tf.get_logger().setLevel(logging.ERROR)
except ImportError:
    tf = None


def get_torch_device():
    """Returns the most appropriate PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_tf_device():
    """Returns a string representation of the best TF device."""
    if tf and tf.config.list_physical_devices("GPU"):
        return "/GPU:0"
    return "/device:CPU:0"
