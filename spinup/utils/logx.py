"""

Some simple logging functionality, inspired by rllab's logging.

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""
import json
import joblib
import shutil
import numpy as np
try:
    import tensorflow as tf
except:
    tf = None
import torch
import os.path as osp, time, atexit, os
import warnings
from spinup.utils.mpi_tools import proc_id, mpi_statistics_scalar
from spinup.utils.serialization_utils import convert_json

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    Admitted colors: black, red, green, yellow, blue, magenta, cyan, white.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def restore_tf2_model(fpath):
    """
    Loads a TF2 model saved by Logger.
    
    Args:
        fpath: Filepath to the tf2_save directory (containing the model folder).
        
    Returns:
        The loaded Keras model.
    """
    if tf is None:
        raise ImportError("TensorFlow is not installed or could not be loaded.")
    return tf.keras.models.load_model(fpath, compile=False)


class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        """
        Args:
            output_dir (string): A directory for saving results to. If 
                None, defaults to a temp directory of the form
                ``/tmp/experiments/s12345``.

            output_fname (string): The name of the tab-separated-values file 
                (within output_dir) where diagnostics will be saved to. 
                Default is ``progress.txt``.

            exp_name (string): The experiment name. If None, defaults to 
                'unnamed'.
        """
        if proc_id()==0:
            self.output_dir = output_dir or "/tmp/experiments/%i"%int(time.time())
            if osp.exists(self.output_dir):
                print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
            atexit.register(self.output_file.close)
            print(colorize("Logging data to %s"%self.output_file.name, 'cyan', bold=True))
        else:
            self.output_dir = None
            self.output_file = None
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color='cyan'):
        """Print a colorized message to stdout."""
        if proc_id()==0:
            print(colorize(msg, color, bold=True))

    def save_config(self, config):
        """
        Log about hyperparameters.

        Args:
            config (dict): A dictionary of hyperparameters.
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if proc_id()==0:
            output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
            print(colorize("Saving config:\n", 'cyan', bold=True))
            print(output)
            with open(osp.join(self.output_dir, "config.json"), 'w') as f:
                f.write(output)

    def save_state(self, state_dict, itr=None):
        """
        Saves the state of an experiment.

        To be clear: this is about saving *the entire state of training*, 
        including the model, the optimizer, the number of epochs completed 
        and so on, so that training can be resumed exactly where it left off.

        Args:
            state_dict (dict): Dictionary containing everything you want to
                save.
            itr (int): Iteration number (optional). If provided, a suffix of 
                that number will be added to the filename of the state save 
                and the model save.
        """
        if proc_id()==0:
            fname = 'vars.pkl' if itr is None else 'vars%d.pkl'%itr
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, fname))
            except:
                self.log('Warning: could not pickle state_dict.', color='red')
            if hasattr(self, 'tf2_saver_elements'):
                self._tf2_simple_save(itr)
            if hasattr(self, 'pytorch_saver_elements'):
                self._pytorch_simple_save(itr)

    def setup_tf2_saver(self, what_to_save):
        """
        Set up easy model saving for a single TF2 model.

        Args:
            what_to_save: Any TF2 model or serializable object containing
                TF2 models.
        """
        self.tf2_saver_elements = what_to_save

    def _tf2_simple_save(self, itr=None):
        """
        Saves the TF2 model.
        """
        if proc_id()==0:
            assert hasattr(self, 'tf2_saver_elements'), \
                "First have to setup saving with self.setup_tf2_saver"
            fpath = 'tf2_save'
            fpath = osp.join(self.output_dir, fpath)
            fname = 'model' + ('%d'%itr if itr is not None else '')
            fname = osp.join(fpath, fname)
            os.makedirs(fpath, exist_ok=True)
            self.tf2_saver_elements.save(fname)

    def setup_pytorch_saver(self, what_to_save):
        """
        Set up easy model saving for a single PyTorch model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; it's really just for prologue.

        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None):
        """
        Saves the PyTorch model (or models).
        """
        if proc_id()==0:
            assert hasattr(self, 'pytorch_saver_elements'), \
                "First have to setup saving with self.setup_pytorch_saver"
            fpath = 'pyt_save'
            fpath = osp.join(self.output_dir, fpath)
            fname = 'model' + ('%d'%itr if itr is not None else '') + '.pt'
            os.makedirs(fpath, exist_ok=True)
            torch.save(self.pytorch_saver_elements, osp.join(fpath, fname))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Args:
            key (string): The name of the diagnostic.
            val: A value for the diagnostic.
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to log a new key %s after first row!"%key
        assert key not in self.log_current_row, "You already set %s this row!"%key
        self.log_current_row[key] = val

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        if proc_id()==0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15,max(key_lens))
            keystr = '%'+'%d'%max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-" * n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g"%val if hasattr(val, "__float__") else val
                print(fmt%(key, valstr))
                vals.append(val)
            print("-" * n_slashes, flush=True)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers)+"\n")
                self.output_file.write("\t".join([str(v) for v in vals])+"\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some diagnostic which is updated many times
    during an epoch, and at the end of the epoch, you want to report the 
    average / std / min / max value of that diagnostic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_dict.
        """
        for k,v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key,val)
        else:
            v = self.epoch_dict[key]
            if len(v) == 0:
                return
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else 'Average' + key, stats[0])
            if not(average_only):
                super().log_tabular('Std'+key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max'+key, stats[3])
                super().log_tabular('Min'+key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        if len(v) == 0:
            return None
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        return mpi_statistics_scalar(vals)
