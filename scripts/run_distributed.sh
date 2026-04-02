#!/bin/bash
#SBATCH --account=[ALLOCATION ACCT]   # <- This is our ACCESS account 
#SBATCH --partition=gpuA40x4          # <- This is the type of node we would like to use 
#SBATCH --job-name=chexpert_train     # <- The name of our job 
#SBATCH --nodes=1                     # <- The number of nodes we want 
#SBATCH --ntasks-per-node=1           # <- We want to run 1 process per node (PyTorch controls the subprocesses on each node)
#SBATCH --gpus-per-node=4             # <- How many GPUs we want per node 
#SBATCH --cpus-per-task=32            # <- How many CPUs we want per node (DataLoader num_workers * gpus_per_node is rule of thumb)
#SBATCH --mem=64G                     # <- How much RAM per node 
#SBATCH --time=0:15:00                # <- How long our process can run for 
#SBATCH --output=logs/%j.out          # <- Where terminal output goes 
#SBATCH --error=logs/%j.err           # <- Where error output goes 

# Load the most up-to-date Python verson on Delta 
module load python

# Load virtual environment containing required packages 
source /projects/bfep/lewis1/venv/bin/activate

# Determine the IP of the control node for SLURM process 
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=25827

# Run distributed training 
srun torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    distributed_train.py --fname config/distributed_config.yaml