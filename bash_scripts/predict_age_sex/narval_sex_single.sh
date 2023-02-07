#!/bin/bash
#SBATCH --job-name=single_sex
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=10:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH --array=0-7
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/single_sex_%a.out

#['ds000030', 'Cardiff', 'UKBB', 'BC', 'UCLA', 'SFARI', 'ABIDE','Orban', 'ADHD200', 'ABIDE2']
# Exclude ABIDE & ABIDE2 - all male
TASK_ARRAY=('ds000030' 'Cardiff' 'UKBB' 'BC' 'UCLA' 'SFARI' 'Orban' 'ADHD200')
task=${TASK_ARRAY[$SLURM_ARRAY_TASK_ID]}

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results/sex/'

# SCRIPT
hps_conf='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/hps_conf.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'Single task MLP on conn predicting SEX in '$task
mkdir $p_out_parent$task
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent$task'/fold_'$fold
    echo $p_out
    mkdir $p_out

    python $hps_conf --tasks $task --type 'conn' --conf 'SEX' --num_epochs 100 --batch_size 8 --encoder 3 --head 3 --data_format 0 --log_dir $p_out --data_dir $data_dir
done