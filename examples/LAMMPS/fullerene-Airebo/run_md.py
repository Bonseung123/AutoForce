import os
import sys
import json 

import torch
from ase.io import read, Trajectory
from theforce.calculator.active import ActiveCalculator
from theforce.util.parallel import mpi_init, rank
from theforce.util.aseutil import init_velocities
from ase import units

import numpy as np 
from ase_md_npt import NPT3
from ase_md_logger import MDLogger3
from ase.calculators.lammpslib import LAMMPSlib
from ase.parallel import parprint

# --- 1. Initialize MPI for parallel execution ---
process_group = mpi_init()
is_master = rank() == 0

# --- 2. Define atoms object ---
# Read the final state from the initialization run to ensure a smooth continuation.
# In a parallel run, ase.io.read is a collective operation. The master reads
# and broadcasts the data, so all processes must call this.
init_traj_file = 'init.traj'
if os.path.exists(init_traj_file):
    atoms = read(init_traj_file, index=-1)
    parprint(f"Continuing MD simulation from the last frame of {init_traj_file}")
else:
    # Fallback to initial structure if init.traj doesn't exist
    atoms = read('fullerene.xyz', index=0)
    parprint(f"Warning: {init_traj_file} not found. Starting from fullerene.xyz.")
    # Initialize velocities if starting from scratch
    temperature_K = 300.0
    init_velocities(atoms, temperature_K)

# --- 3. Setup the LAMMPS calculator ---
# Each process running LAMMPSlib will try to create a log file.
# To prevent conflicts, we give each process a unique log file name.
cmds = ["pair_style airebo 6 0 0",
        "pair_coeff * * CH.airebo   C"]
lmp_calc = LAMMPSlib(lmpcmds=cmds, log_file=f'test_{rank()}.log')

kernel_kw = {'lmax':3, 'nmax':3, 'exponent':4, 'cutoff':6.0}
calc = ActiveCalculator(covariance='pckl',
                           kernel_kw=kernel_kw,
                           calculator=lmp_calc,
                           ediff=0.04,
                           fdiff=0.04,
                           process_group=process_group,
                           pckl="model.pckl",
                           tape="model.sgpr",
                           max_data=50,
                           max_inducing=1000,
                           nbeads=1)

atoms.calc = calc

# --- 4. MD parameters ---
dt_fs = 1.0*units.fs
ttime = 25.0*units.fs
ptime = 100.0*units.fs
bulk_modulus = 137.0
pfactor = (ptime**2)*bulk_modulus * units.GPa
temperature_K = 300.0
temperature = temperature_K * units.kB
external_stress = 0.01 * units.GPa 

# Velocities should be present from the init.traj file.
# If not, they were initialized in the fallback case above.
fixed_temperature = True
fixed_pressure = False

if not fixed_temperature:
    ttime=None

if not fixed_pressure:
    pfactor = None 

anisotropic = False

# --- 5. Setup and Run Dynamics ---
# The custom NPT3 dynamics class requires all processes to participate in the
# simulation loop to synchronize state via MPI broadcasts. This is non-standard
# for ASE. We work around this by running the dynamics on all processes, but
# ensuring that file I/O (logging and trajectory writing) is only performed
# by the master process to prevent file corruption.
dyn = NPT3 (atoms,
            dt_fs,
            temperature=temperature, 
            externalstress=external_stress,
            ttime=ttime,
            pfactor=pfactor,
            anisotropic=anisotropic,
            trajectory=None,  # Disable automatic trajectory writing by NPT3
            logfile=None,     # Disable automatic log writing by NPT3
            loginterval=10)

# Attach loggers and trajectory writers only on the master process
if is_master:
    parprint("Running MD on all processes, with I/O handled by master.")
    logger = MDLogger3(dyn=dyn, atoms=atoms, logfile='md.dat', stress=True)
    # Manually create trajectory on master, append if it exists
    traj = Trajectory('md.traj', 'a', atoms)
    dyn.attach(logger, interval=2)
    dyn.attach(traj.write, interval=dyn.loginterval)

dyn.run (1000000)
