import os
import subprocess
import torch
import sys
import site

# Suppress TensorFlow logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    import tensorflow as tf
except ImportError:
    tf = None

def setup_tf_gpu():
    """Configures environment for TF GPU support if needed."""
    if tf and not tf.config.list_physical_devices('GPU'):
        # Find the active site-packages directory
        active_site_packages = site.getsitepackages()
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            active_site_packages.append(user_site)
        
        lib_dirs = ["/usr/lib/x86_64-linux-gnu"]
        home = os.path.expanduser("~")
        
        # Search for 'nvidia' folder in all potential site-packages
        nvidia_path = None
        for sp in active_site_packages:
            candidate = os.path.join(sp, "nvidia")
            if os.path.exists(candidate):
                nvidia_path = candidate
                break
        
        if nvidia_path:
            for root, dirs, files in os.walk(nvidia_path):
                if 'lib' in dirs:
                    lib_dirs.append(os.path.join(root, 'lib'))
            
            if lib_dirs:
                # Create a temporary directory for version-mapped symlinks
                tf_gpu_libs = os.path.join(home, "tf_gpu_libs")
                os.makedirs(tf_gpu_libs, exist_ok=True)
                
                # Mapping of expected TF 2.15 versions to available versions
                mappings = {
                    'cudnn/lib/libcudnn.so.9': 'libcudnn.so.8',
                    'cublas/lib/libcublas.so.12': 'libcublas.so.11',
                    'cublas/lib/libcublasLt.so.12': 'libcublasLt.so.11',
                    'cusparse/lib/libcusparse.so.12': 'libcusparse.so.11',
                    'cufft/lib/libcufft.so.11': 'libcufft.so.10',
                }
                
                for src_sub, dest_name in mappings.items():
                    # Search for src in all discovered lib_dirs
                    for lib_dir in lib_dirs:
                        if "nvidia" in lib_dir:
                            possible_src = os.path.join(os.path.dirname(lib_dir), os.path.basename(src_sub))
                            if os.path.exists(possible_src):
                                dest = os.path.join(tf_gpu_libs, dest_name)
                                if not os.path.exists(dest):
                                    os.symlink(possible_src, dest)
                                break
                
                lib_dirs.insert(0, tf_gpu_libs)
                current_ld = os.environ.get('LD_LIBRARY_PATH', '')
                os.environ['LD_LIBRARY_PATH'] = ":".join(lib_dirs) + (":" + current_ld if current_ld else "")
                
                # Note: On some systems, changing LD_LIBRARY_PATH after the process starts 
                # might not work for the current process, but it works for subprocesses
                # (like those launched by ExperimentGrid/MPI).

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
