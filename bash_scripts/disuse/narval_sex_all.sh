#!/bin/bash
#SBATCH --job-name=sex_all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=20:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/sex_all.out

# ['ds000030', 'Cardiff', 'UKBB', 'BC', 'UCLA', 'SFARI', 'ABIDE','Orban', 'ADHD200', 'ABIDE2']
# Exclude ABIDE & ABIDE2 - all male
tasks='ds000030 Cardiff UKBB BC UCLA SFARI Orban ADHD200'

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results/sex/all/'

# SCRIPT
hps_conf='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/hps_conf.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'All tasks MLP on conn predicting SEX'
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent'fold_'$fold
    echo $p_out
    mkdir $p_out
   
    python $hps_conf --tasks $tasks --type 'conn' --conf 'SEX' --num_epochs 100 --batch_size 8 --encoder 3 --head 3 --data_format 0 --log_dir $p_out --data_dir $data_dir
done