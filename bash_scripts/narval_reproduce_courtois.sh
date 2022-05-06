#!/bin/bash
#SBATCH --job-name=reproduce_courtois
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=10:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/courtois.out

tasks='SZ ASD BIP DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2 DEL1q21_1 DUP1q21_1'

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/results/reproduce_courtois/'

# SCRIPT
hps='/home/harveyaa/projects/def-pbellec/harveyaa/miniMTL/examples/hps.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'Attempting to reproduce courtois results...'
mkdir $p_out_parent
for run in 0 1 2 3 4
do
    p_out=$p_out_parent'/run_'$run
    echo $p_out
    mkdir $p_out

    python $hps --tasks $tasks --num_epochs 200 --batch_size 16 --encoder 0 --head 0 --data_format 1 --log_dir $p_out --data_dir $data_dir
done