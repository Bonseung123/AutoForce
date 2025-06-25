#!/bin/bash

# This script runs the on-the-fly MLMD simulation in parallel using mpirun.
# It executes the run_md.py script, which is set up to handle MPI parallelism
# for both theforce (the ML part) and LAMMPSlib (the ground truth calculator).
#
# Usage:
# ./run.sh [number_of_processes]
#
# Example: to run on 8 processes
# ./run.sh 8

# Make the script executable first: chmod +x run.sh

# Number of processes to use. Default to 4 if not provided.
NPROCS=${1:-4}

echo "Running on-the-fly MLMD with LAMMPS (Airebo) on $NPROCS processes..."

mpirun -np $NPROCS python run_md.py

echo "Simulation finished."