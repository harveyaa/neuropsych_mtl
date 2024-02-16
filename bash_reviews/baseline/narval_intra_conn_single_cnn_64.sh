#!/bin/bash
#SBATCH --job-name=cnn_64_single
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=15:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH --array=0-10
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/cnn_64_single_%a.out

# NEGATIVE TRANSFER STUDY - CNN BASELINE - paper version (intra-site CV)

TASK_ARRAY=('SZ' 'ASD' 'BIP' 'DEL22q11_2' 'DUP22q11_2' 'DEL16p11_2' 'DUP16p11_2' 'DEL1q21_1' 'DUP1q21_1' 'ADHD' 'DEL15q11_2')
task=${TASK_ARRAY[$SLURM_ARRAY_TASK_ID]}

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'
id_dir='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/datasets/cv_folds/intrasite/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_reviews/baseline/cnn_64/'

# SCRIPT
hps_balanced='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/hps_balanced.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env2/bin/activate

echo 'Single task CNN on conn w/ intrasite CV on '$task
mkdir $p_out_parent$task
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent$task'/fold_'$fold
    echo $p_out
    mkdir $p_out

    python $hps_balanced --tasks $task --type 'conn' --strategy 'balanced' --num_epochs 100 --batch_size 8 --encoder 100 --head 0 --data_format 2 --log_dir $p_out --id_dir $id_dir --data_dir $data_dir --fold $fold
done