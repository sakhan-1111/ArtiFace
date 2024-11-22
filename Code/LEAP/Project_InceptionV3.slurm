#!/bin/bash

#SBATCH --job-name=InceptionV3
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=128Gb
#SBATCH --time=366:59:59
#SBATCH --mail-type=end
#SBATCH --mail-user=email@txstate.edu

echo ""
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

# Move to the correct directory

cd /home/Project
echo "Current working directory is `pwd`"

# Train & Test the model
python Project_InceptionV3.py

# end of the program
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
