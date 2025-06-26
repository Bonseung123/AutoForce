import os
import sys
import json

import torch
from ase.io import read
from theforce.calculator.active import ActiveCalculator
from theforce.util.aseutil import init_velocities
from ase import units

import numpy as np
from ase_md_npt import NPT3
from ase_md_logger import MDLogger3
from ase.calculators.lammpslib import LAMMPSlib

# This script runs in serial to create an initial ML model.
print("Running single-threaded initialization...")

# --- 1. Setup for serial execution ---
process_group = None

# --- 2. Define atoms object ---
atoms = read('fullerene.xyz', index=0)

# --- 3. Setup LAMMPS calculator for serial execution ---
cmds = ["pair_style airebo 6 0 0",
        "pair_coeff * * CH.airebo   C"]
lmp_calc = LAMMPSlib(lmpcmds=cmds, log_file='init.log')

# --- 4. Setup ActiveCalculator to build a new model ---
# Start with an empty model by setting covariance=None.
# The output model will be saved in model.pckl/
kernel_kw = {'lmax': 3, 'nmax': 3, 'exponent': 4, 'cutoff': 6.0}
calc = ActiveCalculator(covariance=None,  # Start from scratch
                        kernel_kw=kernel_kw,
                        calculator=lmp_calc,
                        ediff=0.04,
                        fdiff=0.04,
                        process_group=process_group,
                        pckl="model.pckl",  # Save the generated model here
                        tape="model.sgpr",
                        max_data=50,
                        max_inducing=1000,
                        nbeads=1)

atoms.calc = calc

# --- 5. MD parameters for initialization ---
dt_fs = 1.0 * units.fs
ttime = 25.0 * units.fs
temperature_K = 300.0
temperature = temperature_K * units.kB

init_velocities(atoms, temperature_K)

# --- 6. Run a short NVT simulation to generate data ---
# We use NVT (pfactor=None) for simplicity during initialization.
dyn = NPT3(atoms,
           dt_fs,
           temperature=temperature,
           externalstress=0.0,
           ttime=ttime,
           pfactor=None,  # This makes it NVT
           anisotropic=False,
           trajectory='init.traj',
           logfile=None,
           append_trajectory=False,
           loginterval=10)

logger = MDLogger3(dyn=dyn, atoms=atoms, logfile='init_md.dat', stress=False)
dyn.attach(logger, 2)

# Run for a small number of steps to build an initial model
# This will trigger a few DFT calculations and populate the ML model.
INIT_STEPS = 20
print(f"Running for {INIT_STEPS} steps to create an initial ML model...")
dyn.run(INIT_STEPS)

print("Initialization finished. Model saved in model.pckl/, trajectory in init.traj")