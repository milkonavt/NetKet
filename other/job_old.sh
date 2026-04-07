#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu_test
#SBATCH --mem-per-gpu=30G
#SBATCH -t 0-00:10:00
#SBATCH --output=/n/home03/onikolaenko/NetKet/Kondo_model/logs/slurm-%j.out


J2=0.5

DATA=/n/home03/onikolaenko/NetKet/Kondo_model/data/
SCRDIR=/scratch/$USER/$SLURM_JOBID 
program_to_run=kondo_run.py
outfile_run=info_J2$J2

mkdir --parents $SCRDIR 
mkdir $SCRDIR/data/


module load python/3.12.11-fasrc01
source ~/venvs/netket_developer/bin/activate



cp $program_to_run $SCRDIR
cp kondo_hamiltonian.py $SCRDIR
cp Embedding.py $SCRDIR
cp sampler_rules.py $SCRDIR

cd $SCRDIR

srun python -u $program_to_run $J2>>$DATA/$outfile_run


rm $SCRDIR/$program_to_run
rm $SCRDIR/kondo_hamiltonian.py
rm $SCRDIR/Embedding.py
rm $SCRDIR/sampler_rules.py
rm -rf $SCRDIR 



