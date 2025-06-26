#!/bin/bash

# This script runs the on-the-fly MLMD simulation.
# It first runs a short, single-threaded initialization to create a
# preliminary ML model. Then, it runs the main MD simulation in parallel
# using mpirun.
#
# Usage:
# ./run.sh [number_of_processes]
#
# Example: to run on 8 processes
# ./run.sh 8
# Make the script executable first: chmod +x run.sh

# Number of processes to use for the parallel part. Default to 4 if not provided.
NPROCS=${1:-4}

# --- Step 1: Initialization (single-threaded) ---
# This step creates an initial ML model by running a few MD steps.
# The model and trajectory are saved for the next step.
echo "Running single-threaded initialization (see init_model.py)..."
python init_model.py
if [ $? -ne 0 ]; then
    echo "Initialization failed. Aborting."
    exit 1
fi
echo "Initialization successful."

# --- Step 2: Main MD simulation (multi-threaded) ---
echo "Running on-the-fly MLMD with LAMMPS (Airebo) on $NPROCS processes..."
mpirun -np $NPROCS python run_md.py
if [ $? -ne 0 ]; then
    echo "Main MD simulation failed."
    exit 1
fi
echo "Simulation finished."