#!/bin/bash
#SBATCH --time=1:00:00 --ntasks=1 --nodes=1 -p gpu -o log/slurm-%j.out
#SBATCH --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:p100:1

# Note that we do not want to use the node exclusively,
# especially when using nodes with multiple GPUs (on lobos: k40 -- 2 GPUs, pascal -- 4 GPUs).
# OpenMM does all the work on the GPU and utilizes only one GPU per simulation. 
# By requesting only on GPU per job, the rest of the GPUs can be utilized by other jobs.

cd $SLURM_SUBMIT_DIR

# conda activate openmm # use your conda environment here (e.g. 'openmm')
sleep 10

# run simulation and resubmit script
./dyn.py && rflow submit sdyn.sh && sleep 60
