#!/bin/bash
#SBATCH --job-name=pairs_sm_big_intra
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=12:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH --array=1-55
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/pair_sm_big_%a.out

# NEGATIVE TRANSFER STUDY - Shared Middle deeper PAIRWISE - paper version (intra-site CV)

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'
id_dir='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/datasets/cv_folds/intrasite/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_paper/pairs/mps_big/'

# SCRIPT
mps='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/mps.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

################
# PAIRED TASKS #
################
tasks="$(sed "${SLURM_ARRAY_TASK_ID}q;d" unique_pairs.txt)"
echo $tasks

echo 'paired task MLP on conn w/ intrasite CV on '$task
mkdir $p_out_parent'pair_'$SLURM_ARRAY_TASK_ID
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent'pair_'$SLURM_ARRAY_TASK_ID'/fold_'$fold
    echo $p_out
    mkdir $p_out

    python $mps --tasks $tasks --type 'conn' --num_epochs 100 --batch_size 8 --preencoder 333 --encoder 333 --head 3 --data_format 0 --log_dir $p_out --id_dir $id_dir --data_dir $data_dir --fold $fold
done