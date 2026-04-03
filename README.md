# NAIRR ACCESS-DELTA CheXPert Workshop Demo

These files run a training of ResNet-101 on the CheXpert dataset in a HPC environment. Once logged into Delta, simply run 
`sh scripts/run_distributed.sh` to start the model training. Users will need to specify their own Delta/ACCESS account name  
inside this script. We should also have users all use different values for the MASTER_PORT in this script so that one 
port doesn't become overloaded. 

After cloning this repo on Delta, users should create a directory called `logs` and `runs` inside of the repo directory. These directories will hold trainnig output, but should not be stored in Git. The `.gitignore` files specifies to not include these directories when pushing to this repository. 

Need to ensure that everyone can access the data and virtual environment. 
- Data is in /work/hdd/[ALLOCATION_ID]/lewis1
- Virtual env. is in /work/hdd/[ALLOCATION_ID]/lewis1

Based on my testing, an epoch of CheXpert training on MARCI takes ~50m. This is with the CheXpert data on the 
slow HDD that we have on MARCI. Since we are supposed to put our data in the /work/hdd/ drive on Delta, I think 
we may experience similar data loading bottlenecks. As of this update (4/2 AM), I have a run of this code 
submitted on Delta to gauge the runtime there, but the job has not started running yet. 

I use the Adam optimizer with the LR and betas specified in the [CheXpert paper](https://arxiv.org/pdf/1901.07031).
The authors only train for 3 epochs at a batch size of 16 in this paper. We should plan on increasing this batch size 
to show the benefits of a HPC system.  

-------------------------------------------------------------------------------------------------------------------------------------------

## HPC Overview 

Modern neural networks, particularly those designed to analyze chest X-rays and other forms of medical images, require enormous amounts of computational power to train effectively. A standard desktop or laptop computer, even one equipped with a graphics card, is simply not powerful enough to train these models in a reasonable amount of time. Instead, researchers rely on High-Performance Computing (HPC) systems. These are large clusters of interconnected computers, each equipped with multiple specialized processors called GPUs (Graphics Processing Units), which are designed to perform the kinds of mathematical operations required for training AI models. A single HPC cluster may contain hundreds or thousands of GPUs spread across dozens of machines, collectively providing far more computational power than any individual workstation would be able to.

<img width="600" height="4000" alt="image" src="https://github.com/user-attachments/assets/12306194-6fd4-4ff0-8b9b-4b455fecf1a9" /> <br/>
[1] [https://learn.ncsa.illinois.edu/](https://learn.ncsa.illinois.edu/)


## SLURM Overview
Because these resources are shared among many users simultaneously, HPC systems use a scheduling system to manage who gets access to what hardware and when. The most widely used scheduler in academic and medical research settings is called SLURM (Simple Linux Utility for Resource Management). Rather than running a program directly, a researcher submits a job to SLURM — a script that describes what computational resources are needed (how many machines, how many GPUs, how much memory, and for how long) alongside the actual command to run. SLURM queues these jobs and allocates resources as they become available, ensuring fair access across all users of the system. This means a researcher may submit a job and wait minutes or hours before it begins running, depending on how busy the system is.

When training AI models across GPUs machines simultaneously, known as distributed training, the workload is split across all available GPUs in a coordinated way. Each GPU processes a different subset of the training data, and the results are synchronized across all GPUs after each training step. This coordination is handled by a communication framework called `NCCL`, which is optimized for fast data transfer between GPUs both within a single machine and across multiple machines connected by a high-speed network. From the user's perspective, this process is largely automated. In our model training pipeline, a tool called `torchrun` is responsible for launching the correct number of coordinated training processes across all machines, with SLURM ensuring those machines are reserved and available when the job starts. The following is an example of a SLURM configuration file: 

```bash
#!/bin/bash
#SBATCH --job-name=slurm_example      # <- The name of our job 
#SBATCH --nodes=1                     # <- The number of nodes we want 
#SBATCH --ntasks-per-node=1           # <- We want to run 1 process per node (PyTorch controls the subprocesses on each node)
#SBATCH --gpus-per-node=4             # <- How many GPUs we want per node 
#SBATCH --cpus-per-task=32            # <- How many CPUs we want per node (DataLoader num_workers * gpus_per_node is rule of thumb)
#SBATCH --mem=64G                     # <- How much RAM per node 
#SBATCH --time=1:00:00                # <- How long our process can run for (1 hr in this example)
#SBATCH --output=logs/%j.out          # <- Where standard output goes 
#SBATCH --error=logs/%j.err           # <- Where error output goes 

# Determine the IP of the control node for SLURM process and assign port 
# for the master node of the job 
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=12345

# Run distributed training 
srun torchrun \
    --nproc_per_node=4 \              #  <- Numebr of torch proccess per node (less than or equal to the number of GPUs on node)
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py --fname config.yaml
```

Because of this infrastructure, a model which might take weeks to train on a single GPU can be trained in hours on a modern HPC cluster. For medical AI systems such as automated chest X-ray interpretation, this scalability is critical. These models are typically trained on hundreds of thousands of images, and meaningful results require many passes through the entire dataset. The combination of SLURM for resource management and distributed training frameworks for parallelization have enabled researchers to develop and test models at an incredible pace.

## Delta HPC Overview

Delta is a HPC system developed and managed by Hewlett Packard Enterprise (HPE) and the National Center for Supercomputing Applications (NCSA) at the University of Illinois Urbana-Champaign. The [Delta user guide](https://docs.ncsa.illinois.edu/systems/delta/en/latest/index.html) provides a fantastic overview of this system. NCSA also offers plentiful resrouces for learning about HPC systems [here](https://learn.ncsa.illinois.edu/). 

To train a model using PyTorch code on Delta, a SLURM configuration file very similar to the example above is used. An example of a SLURM configuration file that would be used on Delta is given below. The SLURM configuration for our example is `scripts/run_distributed.sh`. 


```bash
#!/bin/bash
#SBATCH --account=[ALLOCATION ACCT]   # <- Specify your ACCESS account 
#SBATCH --partition=gpuA40x4          # <- Specify the type of node we would like to use 
#SBATCH --job-name=chexpert_train     # <- The name of our job 
#SBATCH --nodes=1                     # <- The number of nodes we want 
#SBATCH --ntasks-per-node=1           # <- We want to run 1 process per node (PyTorch controls the subprocesses on each node)
#SBATCH --gpus-per-node=4             # <- How many GPUs we want per node 
#SBATCH --cpus-per-task=32            # <- How many CPUs we want per node (DataLoader num_workers * gpus_per_node is rule of thumb)
#SBATCH --mem=64G                     # <- How much RAM per node 
#SBATCH --time=1:00:00                # <- How long our process can run for 
#SBATCH --output=logs/%j.out          # <- Where terminal output goes 
#SBATCH --error=logs/%j.err           # <- Where error output goes 

# Load the most up-to-date Python verson on Delta 
module load python

# Load virtual environment containing packages required
# for training pipeline  
source ~/venv/bin/activate

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
```
After specifying the SLURM configuration, jobs are submitted by running `sbatch \[slurm_file_name\]`. 

The 'currency' used to access Delta's compute resources are credits allocated through the [NSF's ACCESS program](https://access-ci.org/). These credits can be exchanged for GPU hours on Delta. GPU hours are the unit of time used to measure the amount of Delta resources used. GPU hours are the number of hours a job on Delta runs for multiplied by the number of GPUs used by the job. 1 Delta GPU hour costs (66.67 * `charge_factor`) ACCESS credits. The `charge_factor` value depends on the type of node being used by a job. Some types of nodes on Delta are more powerful than others, thus they cost more per GPU hour. Our training pipeline uses the `gpuA40x4` nodes which have a `charge_factor of 0.5. These nodes each have 4 NVIDIA A40 GPUs. 
