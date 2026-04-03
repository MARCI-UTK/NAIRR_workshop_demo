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

<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/12306194-6fd4-4ff0-8b9b-4b455fecf1a9" />
\[1\] [https://learn.ncsa.illinois.edu/](https://learn.ncsa.illinois.edu/)

Because these resources are shared among many researchers simultaneously, HPC systems use a scheduling system to manage who gets access to what hardware and when. The most widely used scheduler in academic and medical research settings is called SLURM (Simple Linux Utility for Resource Management). Rather than running a program directly, a researcher submits a job to SLURM — a script that describes what computational resources are needed (how many machines, how many GPUs, how much memory, and for how long) alongside the actual command to run. SLURM queues these jobs and allocates resources as they become available, ensuring fair access across all users of the system. This means a researcher may submit a job and wait minutes or hours before it begins running, depending on how busy the cluster is.
When training AI models across multiple machines simultaneously — a technique called distributed training — the model's workload is split across all available GPUs in a coordinated way. Each GPU processes a different subset of the training data in parallel, and the results are synchronized across all GPUs after each training step. This coordination is handled by a communication framework called NCCL, which is optimized for fast data transfer between GPUs both within a single machine and across multiple machines connected by a high-speed network. From the researcher's perspective, this process is largely automated — a tool called torchrun is responsible for launching the correct number of coordinated processes across all machines, with SLURM ensuring those machines are reserved and available when the job starts.
The practical result of this infrastructure is that a model which might take weeks to train on a single GPU can be trained in hours on a modern HPC cluster. For medical AI applications such as automated chest X-ray interpretation, this scalability is critical — models are typically trained on hundreds of thousands of images, and meaningful results require many repeated passes through the entire dataset. The combination of SLURM for resource management, distributed training frameworks for parallelization, and large curated datasets like CheXpert has made it possible for research groups to develop and iterate on clinical AI models at a pace that would have been impossible just a decade ago.
