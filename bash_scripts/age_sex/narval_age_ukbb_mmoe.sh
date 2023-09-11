#!/bin/bash
#SBATCH --job-name=age_ukbb_mmoe
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=160:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH --array=0-59
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/age_ukbb_mmoe_%a.out

# UKBB LEARNING CURVE STUDY - MTL AGE - MMOE

K_ARRAY=(50 100 200 500 1000 1500 2000 2500 3000 3500 4000 4500 50 100 200 500 1000 1500 2000 2500 3000 3500 4000 4500 50 100 200 500 1000 1500 2000 2500 3000 3500 4000 4500 50 100 200 500 1000 1500 2000 2500 3000 3500 4000 4500 50 100 200 500 1000 1500 2000 2500 3000 3500 4000 4500)
k=${K_ARRAY[$SLURM_ARRAY_TASK_ID]}

FOLD_ARRAY=(0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4)
fold=${FOLD_ARRAY[$SLURM_ARRAY_TASK_ID]}

tasks='UKBB11025 UKBB11026 UKBB11027'

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_paper/age/ukbb/mmoe/'

# SCRIPT
mmoe='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/mmoe_conf.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'UKBB sites mtl MMOE on conn predicting AGE'
mkdir $p_out_parent$k

p_out=$p_out_parent$k'/fold_'$fold
echo $p_out
mkdir $p_out
   
python $mmoe --tasks $tasks --type 'conn' --n_subsamp $k --conf 'AGE' --num_epochs 100 --batch_size 8 --encoder 3 --head 3 --num_experts 6 --data_format 0 --fold $fold --log_dir $p_out --data_dir $data_dir