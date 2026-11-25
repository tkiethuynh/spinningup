import numpy as np
import torch
from spinup.utils.mpi_tools import broadcast, mpi_avg, num_procs


def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)


def mpi_avg_grads(module):
    """Average contents of gradient buffers across MPI processes."""
    if num_procs() == 1:
        return
    for p in module.parameters():
        if p.grad is None:
            continue
        p_grad_numpy = np.ascontiguousarray(p.grad.detach().cpu().numpy())
        avg_p_grad = mpi_avg(p_grad_numpy)
        with torch.no_grad():
            p.grad.copy_(torch.as_tensor(avg_p_grad, dtype=p.grad.dtype, device=p.grad.device))


def sync_params(module):
    """Sync all parameters of module across all MPI processes."""
    if num_procs() == 1:
        return
    with torch.no_grad():
        for p in module.parameters():
            p_numpy = np.ascontiguousarray(p.detach().cpu().numpy())
            broadcast(p_numpy)
            p.copy_(torch.as_tensor(p_numpy, dtype=p.dtype, device=p.device))
