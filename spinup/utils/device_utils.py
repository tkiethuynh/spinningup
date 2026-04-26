import os
import subprocess
import torch
import sys
import site

def setup_tf_gpu():
    """
    Configures environment for TF GPU support.
    Uses ctypes to pre-load NVIDIA libraries into the global symbol table,
    solving the LD_LIBRARY_PATH restriction in existing processes.
    """
    # Suppress TensorFlow C++ logging to solve noisy initialization warnings (Factory already registered)
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
        lib_dirs = []
        for root, dirs, files in os.walk(nvidia_path):
            if 'lib' in dirs:
                lib_dirs.append(os.path.join(root, 'lib'))
        
        if lib_dirs:
            current_ld = os.environ.get('LD_LIBRARY_PATH', '')
            os.environ['LD_LIBRARY_PATH'] = ":".join(lib_dirs) + (":" + current_ld if current_ld else "")
            
            # To ensure the current process sees them, use ctypes RTLD_GLOBAL
            import ctypes
            core_libs = [
                'libcudart.so.11.0', 
                'libcublas.so.11', 
                'libcublasLt.so.11', 
                'libcudnn.so.8', 
                'libcufft.so.10', 
                'libcurand.so.10', 
                'libcusolver.so.11', 
                'libcusparse.so.11'
            ]
            
            for lib_name in core_libs:
                for lib_dir in lib_dirs:
                    full_path = os.path.join(lib_dir, lib_name)
                    if os.path.exists(full_path):
                        try:
                            ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                            break
                        except Exception:
                            pass

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

