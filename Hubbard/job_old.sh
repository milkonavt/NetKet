#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu_test
#SBATCH --mem-per-gpu=40G
#SBATCH -t 0-00:30:00
#SBATCH --output=/n/home03/onikolaenko/NetKet/Hubbard/logs/slurm-%j.out

export WANDB_API_KEY="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"
export WANDB_BASE_URL="https://api.wandb.ai"


DATA=/n/home03/onikolaenko/NetKet/Hubbard/data/
SCRDIR=/scratch/$USER/$SLURM_JOBID 
program_to_run=hubbard_run.py
outfile_run=info_file

mkdir --parents $SCRDIR 
mkdir $SCRDIR/data/


module load python/3.12.11-fasrc01
source ~/venvs/netket/bin/activate



cp $program_to_run $SCRDIR 
cd $SCRDIR

srun python -u $program_to_run >>$DATA/$outfile_run


rm $SCRDIR/$program_to_run 
rm -rf $SCRDIR 



