#!/bin/bash
#SBATCH --job-name=all_cnn_intra
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=30:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/all_cnn_intra.out

# CNVs & PSYCH STUDY - MTL CONDITIONS - INTRASITE CV - CNN

tasks='SZ ASD BIP ADHD DEL15q11_2 DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2 DEL1q21_1 DUP1q21_1'

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'
id_dir='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/datasets/cv_folds/intrasite/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_reviews/mtl/cnn/'

# SCRIPT
hps_balanced='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/hps_balanced.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'All tasks CNN'
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent'fold_'$fold
    echo $p_out
    mkdir $p_out

    python $hps_balanced --tasks $tasks --type 'conn' --strategy 'balanced' --num_epochs 100 --batch_size 8 --encoder 0 --head 0 --data_format 1 --log_dir $p_out --id_dir $id_dir --data_dir $data_dir --fold $fold    
done