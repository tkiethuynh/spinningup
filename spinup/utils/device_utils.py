import os
import subprocess
import torch
import sys
import site

def setup_tf_gpu():
    """
    Configures environment for TF GPU support.
    Uses os.execv to restart the process with the correct LD_LIBRARY_PATH,
    solving TensorFlow's strict library loading requirements.
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
            home = os.path.expanduser("~")
            tf_gpu_libs = os.path.join(home, "tf_gpu_libs")
            os.makedirs(tf_gpu_libs, exist_ok=True)
            
            # Mapping of available versions to TF 2.15 expected versions
            mappings = {
                'cuda_runtime/lib/libcudart.so.12': 'libcudart.so.11.0',
                'cuda_runtime/lib/libcudart.so.13': 'libcudart.so.11.0',
                'cublas/lib/libcublas.so.12': 'libcublas.so.11',
                'cublas/lib/libcublas.so.13': 'libcublas.so.11',
                'cublas/lib/libcublasLt.so.12': 'libcublasLt.so.11',
                'cublas/lib/libcublasLt.so.13': 'libcublasLt.so.11',
                'cudnn/lib/libcudnn.so.9': 'libcudnn.so.8',
                'cufft/lib/libcufft.so.11': 'libcufft.so.10',
                'cufft/lib/libcufft.so.12': 'libcufft.so.10',
                'curand/lib/libcurand.so.10': 'libcurand.so.10',
                'cusolver/lib/libcusolver.so.11': 'libcusolver.so.11',
                'cusolver/lib/libcusolver.so.12': 'libcusolver.so.11',
                'cusparse/lib/libcusparse.so.12': 'libcusparse.so.11',
                'cusparse/lib/libcusparse.so.13': 'libcusparse.so.11',
            }
            
            for src_sub, dest_name in mappings.items():
                for lib_dir in lib_dirs:
                    if "nvidia" in lib_dir:
                        possible_src = os.path.join(os.path.dirname(lib_dir), os.path.basename(src_sub))
                        if os.path.exists(possible_src):
                            dest = os.path.join(tf_gpu_libs, dest_name)
                            if not os.path.exists(dest):
                                try:
                                    os.symlink(possible_src, dest)
                                except FileExistsError:
                                    pass
                            break
            
            lib_dirs.insert(0, tf_gpu_libs)
            new_ld = ":".join(lib_dirs)
            current_ld = os.environ.get('LD_LIBRARY_PATH', '')
            
            if "SPINUP_TF_GPU_CONFIGURED" not in os.environ:
                os.environ["SPINUP_TF_GPU_CONFIGURED"] = "1"
                if new_ld not in current_ld:
                    os.environ["LD_LIBRARY_PATH"] = new_ld + (":" + current_ld if current_ld else "")
                    
                    # Ensure python doesn't get confused if run as a module vs script
                    if sys.argv[0] == '-m':
                        pass # Cannot easily execv with -m, fallback to just setting env (unlikely here since spinup/__init__ changes sys.argv)
                    elif sys.argv[0].endswith('pytest'):
                        pass # Don't execv in pytest, let the user set it or rely on fallback
                    else:
                        os.execv(sys.executable, [sys.executable] + sys.argv)

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

