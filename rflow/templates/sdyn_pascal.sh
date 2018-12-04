#!/bin/bash

#SBATCH --mail-type=FAIL,REQUEUE --requeue
#SBATCH --ntasks=1 --nodes=1 -p pascal --qos=veryshort
#SBATCH --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:1

cd $SLURM_SUBMIT_DIR

conda activate openmm-7.4.0_cuda-9.2 # use your conda environment here (e.g. 'openmm')
module load cuda/9.2 # load the appropriate cuda module (e.g. 'cuda/9' for the current openmm version)
sleep 10

# run simulation and resubmit script
./dyn.py && rflow submit sdyn_pascal.sh && sleep 60
