#!/bin/bash
#SBATCH --job-name=mmoe_22
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=40:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/mmoe_22.out

# CNVs & PSYCH STUDY - MTL CONDITIONS - MMOE w/ 22 EXPERTS (double n tasks)

tasks='SZ ASD BIP ADHD DEL15q11_2 DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2 DEL1q21_1 DUP1q21_1'

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'
id_dir='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/datasets/cv_folds/intrasite/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_reviews/mtl/mmoe_22/'

# SCRIPT
mmoe='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/mmoe.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env2/bin/activate

echo 'All tasks MMOE - 22 experts'
# for fold in 0 1 2 3 4
for fold in 3 4
do
    p_out=$p_out_parent'fold_'$fold
    echo $p_out
    mkdir $p_out

    python $mmoe --tasks $tasks --type 'conn' --num_epochs 100 --batch_size 8 --num_experts 22 --encoder 3 --head 3 --data_format 0 --log_dir $p_out --id_dir $id_dir --data_dir $data_dir --fold $fold    
done