# NAIRR Workshop Demo

These files run a training of ResNet-101 on the CheXpert dataset in a HPC environment. Once logged into Delta, simply run `sh scripts/run_distributed.sh` to start the model training. 

After cloning this repo on Delta, users should create a directory called `logs` and `runs` inside of the repo directory. These directories will hold trainnig output, but should not be stored in Git. The `.gitignore` files specifies to not include these directories when pushing to this repository. 

Need to ensure that everyone can access the data and virtual environment. 
- Data is in /work/hdd/[ALLOCATION_ID]/lewis1
- Virtual env. is in /work/hdd/[ALLOCATION_ID]/lewis1
