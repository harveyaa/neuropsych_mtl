#!/bin/bash
#SBATCH --job-name=mmoe_8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=40:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/mmoe_8.out

# PSYCH STUDY - MTL CONDITIONS - MMOE w/ 8 EXPERTS (double n tasks)

tasks='SZ ASD BIP ADHD'

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'
id_dir='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/datasets/cv_folds/intrasite/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_paper/mtl_psych/mmoe_8/'

# SCRIPT
mmoe='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/mmoe.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'Psych tasks MMOE - 8 experts'
# for fold in 0 1 2 3 4
for fold in 0 1 2
do
    p_out=$p_out_parent'fold_'$fold
    echo $p_out
    mkdir $p_out

    python $mmoe --tasks $tasks --type 'conn' --num_epochs 100 --batch_size 8 --num_experts 8 --encoder 3 --head 3 --data_format 0 --log_dir $p_out --id_dir $id_dir --data_dir $data_dir --fold $fold    
done
