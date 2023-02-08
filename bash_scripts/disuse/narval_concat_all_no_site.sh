#!/bin/bash
#SBATCH --job-name=concat_all_no_site
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=15:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/concat_all.out

tasks='SZ ASD BIP DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2 DEL1q21_1 DUP1q21_1'

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'
id_dir='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/datasets/cv_folds/hybrid/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/results/concat_no_site/all/'

# SCRIPT
hps_balanced='/home/harveyaa/projects/def-pbellec/harveyaa/miniMTL/examples/hps_balanced.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'Concat no site - All tasks MLP'
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent'fold_'$fold
    echo $p_out
    mkdir $p_out

    python $hps_balanced --tasks $tasks --type 'concat_no_site' --num_epochs 100 --batch_size 8 --encoder 7 --head 7 --data_format 0 --log_dir $p_out --id_dir $id_dir --data_dir $data_dir --rand_test
done