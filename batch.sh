#!/bin/sh
#
#SBATCH --job-name="sl2rl_job"  
#SBATCH --account=research-3me-dcsc 
#SBATCH --output=logs/trainer_outs_%j.out 
#SBATCH --error=logs/trainer_errs_%j.err 
#SBATCH --partition=gpu  
#SBATCH --time=48:00:00  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB  
#SBATCH --gpus-per-task=0

srun apptainer run --writable-tmpfs --nv --bind $PWD image.sif /bin/bash -c \
    "cd examples/offline_rl && \
     pip install jaxopt && \
     pip install jaxtyping && \
     pip install optax && \
     python trainer.py --exp-name $4 --datasize $2 --device cpu --seed $1 --env-name hopper --algo-name $3"