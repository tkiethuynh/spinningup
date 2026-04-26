import numpy as np
import tensorflow as tf
from mpi4py import MPI
from spinup.utils.mpi_tools import broadcast, mpi_avg, num_procs, proc_id

def sync_params(variables):
    """Sync all variables across MPI processes."""
    if num_procs() == 1:
        return
    for var in variables:
        var_numpy = var.numpy()
        broadcast(var_numpy)
        var.assign(var_numpy)

def mpi_avg_grads(gradients):
    """Average gradients across MPI processes."""
    if num_procs() == 1:
        return gradients
    avg_grads = []
    for grad in gradients:
        if grad is not None:
            avg_grad = mpi_avg(grad.numpy())
            avg_grads.append(tf.convert_to_tensor(avg_grad))
        else:
            avg_grads.append(None)
    return avg_grads

class MpiAdamOptimizer(tf.keras.optimizers.Adam):
    """
    Adam optimizer that averages gradients across MPI processes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        grads, vars = zip(*grads_and_vars)
        avg_grads = mpi_avg_grads(grads)
        return super().apply_gradients(zip(avg_grads, vars), name=name, **kwargs)
