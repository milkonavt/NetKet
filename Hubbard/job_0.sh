#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -p gpu_h200
#SBATCH --mem-per-gpu=240G
#SBATCH -t 0-00:30:00
#SBATCH --output=/n/home03/onikolaenko/NetKet/Hubbard/logs/slurm-%j.out




DATA=/n/home03/onikolaenko/NetKet/Hubbard/data/
SCRDIR=/scratch/$USER/$SLURM_JOBID 
program_to_run=hubbard_benchmark.py
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



