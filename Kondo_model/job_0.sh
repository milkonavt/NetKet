#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -p gpu_h200
#SBATCH --mem-per-gpu=140G
#SBATCH -t 0-02:00:00
#SBATCH --output=/n/home03/onikolaenko/NetKet/Kondo_model/logs/slurm-%j.out

export WANDB_API_KEY="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"
export WANDB_BASE_URL="https://api.wandb.ai"


J2=0.5
PROJECT_DIR=/n/home03/onikolaenko/NetKet/Kondo_model/
DATA=/n/home03/onikolaenko/NetKet/Kondo_model/data/

program_to_run=kondo_run.py
outfile_run=info_J2$J2



module load python/3.12.11-fasrc01
source ~/venvs/netket_developer/bin/activate



cd $PROJECT_DIR

srun python -u $program_to_run $J2>>$DATA/$outfile_run




