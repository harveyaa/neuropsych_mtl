#!/bin/bash
#SBATCH --job-name=mlp_ablation
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=15:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH --array=0-10
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/mlp_ablation_%a.out

# CNVs & PSYCH STUDY - SINGLE TASK CONDITIONS - INTRASITE CV
# NEGATIVE TRANSFER STUDY - MLPconn BASELINE

TASK_ARRAY=('SZ' 'ASD' 'BIP' 'DEL22q11_2' 'DUP22q11_2' 'DEL16p11_2' 'DUP16p11_2' 'DEL1q21_1' 'DUP1q21_1' 'ADHD' 'DEL15q11_2')
skip_task=${TASK_ARRAY[$SLURM_ARRAY_TASK_ID]}

# build the new task list
tasks=''
for i in 0 1 2 3 4 5 6 7 8 9 10
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
id_dir='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/datasets/cv_folds/intrasite/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_reviews/mtl/mlp_ablation/'

# SCRIPT
hps_balanced='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/hps_balanced.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env2/bin/activate

mkdir $p_out_parent$skip_task
#for fold in 0 1 2 3 4
for fold in 4
do
    p_out=$p_out_parent$skip_task'/fold_'$fold
    echo $p_out
    mkdir $p_out

    python $hps_balanced --tasks $tasks --type 'conn' --strategy 'balanced' --num_epochs 100 --batch_size 8 --encoder 3 --head 3 --data_format 0 --log_dir $p_out --id_dir $id_dir --data_dir $data_dir --fold $fold
done