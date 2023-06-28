#!/bin/bash
#SBATCH --job-name=pairs_conn_rand_test
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=5:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH --array=1-36
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/pair_%a.out

# NEGATIVE TRANSFER STUDY - MLPconn PAIRWISE - RAND SPLIT SENSITIVITY ANALSIS

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'
id_dir='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/datasets/cv_folds/hybrid/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results/neg_transfer_rand/mlp/'

# SCRIPT
hps_balanced='/home/harveyaa/projects/def-pbellec/harveyaa/miniMTL/examples/hps_balanced.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

################
# PAIRED TASKS #
################
tasks="$(sed "${SLURM_ARRAY_TASK_ID}q;d" unique_pairs.txt)"
echo $tasks

echo 'paired task MLP on conn w/ random test sets on '$task
mkdir $p_out_parent'pair_'$SLURM_ARRAY_TASK_ID
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent'pair_'$SLURM_ARRAY_TASK_ID'/fold_'$fold
    echo $p_out
    mkdir $p_out

    python $hps_balanced --tasks $tasks --type 'conn' --num_epochs 100 --batch_size 8 --encoder 3 --head 3 --data_format 0 --log_dir $p_out --id_dir $id_dir --data_dir $data_dir --fold $fold --rand_test
done