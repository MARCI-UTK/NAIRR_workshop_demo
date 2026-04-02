# NAIRR Workshop Demo

These files run a training of ResNet-101 on the CheXpert dataset in a HPC environment. Once logged into Delta, simply run 
`sh scripts/run_distributed.sh` to start the model training. 

After cloning this repo on Delta, users should create a directory called `logs` and `runs` inside of the repo directory. These directories will hold trainnig output, but should not be stored in Git. The `.gitignore` files specifies to not include these directories when pushing to this repository. 

Need to ensure that everyone can access the data and virtual environment. 
- Data is in /work/hdd/[ALLOCATION_ID]/lewis1
- Virtual env. is in /work/hdd/[ALLOCATION_ID]/lewis1

Based on my testing, an epoch of CheXpert training on MARCI takes ~1 hr. This is with the CheXpert data on the 
slow HDD that we have on MARCI. Since we are supposed to put our data in the /work/hdd/ drive on Delta, I think 
we may experience similar data loading bottlenecks. As of this update (4/2 AM), I have a run of this code 
submitted on Delta to gauge the runtime there, but the job has not started running yet. 