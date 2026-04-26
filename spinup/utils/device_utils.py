import os
import subprocess
import torch
try:
    import tensorflow as tf
except ImportError:
    tf = None

def setup_tf_gpu():
    """Configures environment for TF GPU support if needed."""
    if tf and not tf.config.list_physical_devices('GPU'):
        # Attempt to find local nvidia libraries and add them to LD_LIBRARY_PATH
        home = os.path.expanduser("~")
        site_packages = os.path.join(home, ".local/lib/python3.10/site-packages")
        nvidia_path = os.path.join(site_packages, "nvidia")
        
        lib_dirs = ["/usr/lib/x86_64-linux-gnu"]
        
        if os.path.exists(nvidia_path):
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
                    # Search for src in all nvidia paths
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
