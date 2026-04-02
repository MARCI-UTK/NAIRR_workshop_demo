#!/bin/bash
#SBATCH --account=bfep-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=chexpert_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8      
#SBATCH --mem=64G
#SBATCH --time=0:15:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load python
source /projects/bfep/lewis1/venv/bin/activate

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=25827

srun torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    distributed_train.py --fname config/distributed_config.yaml