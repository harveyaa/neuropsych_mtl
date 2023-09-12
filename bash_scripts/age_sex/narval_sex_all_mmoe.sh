#!/bin/bash
#SBATCH --job-name=sex_mmoe
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=40:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/sex_mmoe.out

# SIMPLE TARGETS STUDY - MMOE SEX - 28 experts

# sites with at least 30 controls
# Exclude NYU, SZ1, SZ2, USM - insufficient females
tasks='ADHD1 ADHD3 ADHD5 ADHD6 HSJ SZ3 SZ6 Svip1 Svip2 UCLA_CB UCLA_DS1 UKBB11025 UKBB11026 UKBB11027'

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_paper/sex/all_mmoe/'

# SCRIPT
mmoe='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/mmoe_conf.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'All tasks MMOE on conn predicting SEX'
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent'fold_'$fold
    echo $p_out
    mkdir $p_out
   
    python $mmoe --tasks $tasks --type 'conn' --n_subsamp 50 --conf 'SEX' --num_epochs 100 --batch_size 8 --encoder 3 --head 3 --num_experts 28 --data_format 0 --fold $fold --log_dir $p_out --data_dir $data_dir
done