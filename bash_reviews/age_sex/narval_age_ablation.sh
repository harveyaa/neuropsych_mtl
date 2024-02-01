#!/bin/bash
#SBATCH --job-name=single_age
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=15:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH --array=0-17
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/age_ablation_%a.out

# SIMPLE TARGETS STUDY - SINGLE TASK AGE

# sites with at least 30 controls
TASK_ARRAY=('ADHD1' 'ADHD3' 'ADHD5' 'ADHD6' 'HSJ' 'NYU' 'SZ1' 'SZ2' 'SZ3' 'SZ6' 'Svip1' 'Svip2' 'UCLA_CB' 'UCLA_DS1' 'UKBB11025' 'UKBB11026' 'UKBB11027' 'USM')
skip_task=${TASK_ARRAY[$SLURM_ARRAY_TASK_ID]}

# build the new task list
tasks=''
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
do
    var=${TASK_ARRAY[$i]}
    if [ $var != $skip_task ]
        then
        tasks=$tasks' '$var
    fi
done
# cut the leading blank space
tasks="${tasks:1}"
echo 'Ablation study dropping '$skip_task
echo 'TASKS: '$tasks

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_reviews/age/age_ablation/'

# SCRIPT
hps_conf='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/hps_conf.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env2/bin/activate

mkdir $p_out_parent$skip_task
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent$skip_task'/fold_'$fold
    echo $p_out
    mkdir $p_out

    python $hps_conf --tasks $tasks --type 'conn' --n_subsamp 50 --conf 'AGE' --num_epochs 100 --batch_size 8 --encoder 3 --head 33 --data_format 0 --fold $fold --log_dir $p_out --data_dir $data_dir
done
