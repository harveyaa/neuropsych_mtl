#!/bin/bash
#SBATCH --job-name=single_sex_thinner
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=15:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH --array=0-13
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/single_sex_thinner_%a.out

# SIMPLE TARGETS STUDY - SINGLE TASK SEX

# sites with at least 30 controls
# Exclude NYU, SZ1, SZ2, USM - insufficient females
TASK_ARRAY=('ADHD1' 'ADHD3' 'ADHD5' 'ADHD6' 'HSJ' 'SZ3' 'SZ6' 'Svip1' 'Svip2' 'UCLA_CB' 'UCLA_DS1' 'UKBB11025' 'UKBB11026' 'UKBB11027')
task=${TASK_ARRAY[$SLURM_ARRAY_TASK_ID]}

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_reviews/sex/model_vars/thinner/'

# SCRIPT
hps_conf='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/hps_conf.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env2/bin/activate

echo 'Single task MLPconn_thinner on conn predicting SEX in '$task
mkdir $p_out_parent$task
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent$task'/fold_'$fold
    echo $p_out
    mkdir $p_out

    python $hps_conf --tasks $task --type 'conn' --n_subsamp 50 --conf 'SEX' --num_epochs 100 --batch_size 8 --encoder 10 --head 10 --data_format 0 --fold $fold  --log_dir $p_out --data_dir $data_dir
done
