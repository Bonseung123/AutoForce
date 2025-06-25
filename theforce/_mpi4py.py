# +
import sys

import numpy as np
import torch # Added import for torch
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
# if mpi4py is available in sys.modules,
# ase parallelization is activated which
# apparently has conflicts with theforce
# since no ase parallelization is assumed.
# The right way to corrcet the code is
# to search for all ase function invocations
# and consider parallelism case by case.
# For now, the ugly solution is:
# sys.modules['_mpi4py'] = sys.modules['mpi4py']
del sys.modules["mpi4py"]


def is_initialized():
    return True


class group:
    WORLD = comm


class ReduceOp:
    MAX = MPI.MAX
    SUM = MPI.SUM


def init_process_group(arg="mpi"):
    assert arg == "mpi"


def get_world_size(group=comm):
    return size


def get_rank(group=comm):
    return rank


def broadcast(data, src): # Modified function
    # Determine if the original data was a torch.Tensor
    is_torch_tensor = isinstance(data, torch.Tensor)

    # Convert to numpy array for MPI communication
    if is_torch_tensor:
        np_data = data.detach().numpy()
    elif isinstance(data, np.ndarray):
        np_data = data
    else:
        np_data = np.array(data)

    # Ensure the numpy array is contiguous and writable for MPI
    if not np_data.flags['C_CONTIGUOUS'] or not np_data.flags['WRITEABLE']:
        np_data = np.ascontiguousarray(np_data)

    comm.Bcast(np_data, src)

    # Convert back to torch.Tensor if the original data was a torch.Tensor
    if is_torch_tensor:
        return torch.from_numpy(np_data)
    elif isinstance(data, np.ndarray):
        return np_data
    else:
        return np_data.item() # For scalars, return the scalar value


def all_reduce(data, op=ReduceOp.SUM):
    a = data.detach().numpy().reshape(-1)
    b = np.zeros_like(a)
    comm.Allreduce(a, b, op)
    a[:] = b[:]


def barrier():
    comm.Barrier()
