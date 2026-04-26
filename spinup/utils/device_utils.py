import os
import subprocess
import torch
import sys
import site
import ctypes

# Suppress TensorFlow logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    import tensorflow as tf
except ImportError:
    tf = None

def setup_tf_gpu():
    """
    Configures environment for TF GPU support.
    Uses ctypes to pre-load NVIDIA libraries into the global symbol table,
    solving the LD_LIBRARY_PATH restriction in existing processes.
    """
    if tf and not tf.config.list_physical_devices('GPU'):
        # Find the active site-packages directory
        active_site_packages = site.getsitepackages()
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            active_site_packages.append(user_site)
        
        # Search for 'nvidia' folder in all potential site-packages
        nvidia_path = None
        for sp in active_site_packages:
            candidate = os.path.join(sp, "nvidia")
            if os.path.exists(candidate):
                nvidia_path = candidate
                break
        
        if nvidia_path:
            # Order of loading is important for dependencies
            lib_targets = [
                ('cuda_runtime', 'libcudart'),
                ('cublas', 'libcublas'),
                ('cudnn', 'libcudnn'),
                ('cufft', 'libcufft'),
                ('curand', 'libcurand'),
                ('cusolver', 'libcusolver'),
                ('cusparse', 'libcusparse')
            ]
            
            for pkg, lib_name in lib_targets:
                found = False
                pkg_path = os.path.join(nvidia_path, pkg, 'lib')
                if os.path.exists(pkg_path):
                    for f in os.listdir(pkg_path):
                        if f.startswith(lib_name + ".so"):
                            full_path = os.path.join(pkg_path, f)
                            try:
                                # RTLD_GLOBAL is key: it makes symbols available to subsequent loads (like TF)
                                ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                                found = True
                            except Exception as e:
                                pass
                            if found: break

def get_torch_device():
    """Returns the most appropriate PyTorch device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def get_tf_device():
    """Returns a string representation of the best TF device."""
    if tf:
        setup_tf_gpu()
        if tf.config.list_physical_devices('GPU'):
            return '/GPU:0'
    return '/device:CPU:0'

# Call setup BEFORE importing tensorflow
setup_tf_gpu()

try:
    import tensorflow as tf
except ImportError:
    tf = None

def get_torch_device():
    """Returns the most appropriate PyTorch device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def get_tf_device():
    """Returns a string representation of the best TF device."""
    if tf and tf.config.list_physical_devices('GPU'):
        return '/GPU:0'
    return '/device:CPU:0'
