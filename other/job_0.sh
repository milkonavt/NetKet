#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu_test
#SBATCH --mem-per-gpu=30G
#SBATCH -t 0-00:10:00
#SBATCH --output=/n/home03/onikolaenko/NetKet/Kondo_model/logs/slurm-%j.out

J2=0.5

PROJECT_DIR=/n/home03/onikolaenko/NetKet/Kondo_model
SCRATCH_BASE=/n/netscratch/$USER
SCRDIR=$SCRATCH_BASE/$SLURM_JOBID

mkdir -p $SCRDIR

module load python/3.12.11-fasrc01
source ~/venvs/netket_developer/bin/activate

cd $PROJECT_DIR

# run (write to scratch)
srun python -u kondo_run.py $J2 --outdir $SCRDIR

# copy results back
cp -r $SCRDIR/* $PROJECT_DIR/data/

# clean up
rm -rf $SCRDIR