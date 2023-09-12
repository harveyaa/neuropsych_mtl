#!/bin/bash
#SBATCH --job-name=pairs_mmoe
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=20:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH --array=1-55
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/pair_mmoe_%a.out

# NEGATIVE TRANSFER STUDY - MMOE PAIRWISE - 4 experts - paper version (intra-site CV)

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'
id_dir='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/datasets/cv_folds/intrasite/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_paper/pairs/mmoe/'

# SCRIPT
mmoe='/home/harveyaa/projects/def-pbellec/harveyaa/miniMTL/examples/mmoe.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

################
# PAIRED TASKS #
################
tasks="$(sed "${SLURM_ARRAY_TASK_ID}q;d" unique_pairs.txt)"
echo $tasks

echo 'paired task MMOE on conn w/ intrasite CV '$task
mkdir $p_out_parent'pair_'$SLURM_ARRAY_TASK_ID
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent'pair_'$SLURM_ARRAY_TASK_ID'/fold_'$fold
    echo $p_out
    mkdir $p_out

    python $mmoe --tasks $tasks --type 'conn' --strategy 'balanced' --num_epochs 100 --batch_size 8 --encoder 3 --head 3 --num_experts 4 --data_format 0 --log_dir $p_out --id_dir $id_dir --data_dir $data_dir --fold $fold
done