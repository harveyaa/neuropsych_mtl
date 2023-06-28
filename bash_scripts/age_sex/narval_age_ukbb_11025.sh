#!/bin/bash
#SBATCH --job-name=age_ukbb_11025
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=150:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH --array=0-44
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/age_ukbb_single_%a.out

# UKBB LEARNING CURVE STUDY - SINGLE AGE

K_ARRAY=(150 300 600 6000 7500 9000 10500 12000 13500 150 300 600 6000 7500 9000 10500 12000 13500 150 300 600 6000 7500 9000 10500 12000 13500 150 300 600 6000 7500 9000 10500 12000 13500 150 300 600 6000 7500 9000 10500 12000 13500)
k=${K_ARRAY[$SLURM_ARRAY_TASK_ID]}

FOLD_ARRAY=(0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4)
fold=${FOLD_ARRAY[$SLURM_ARRAY_TASK_ID]}

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_paper/age/ukbb/ukbb_11025/'

# SCRIPT
hps_conf='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/hps_conf.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'UKBB sites mtl MLP on conn predicting AGE'
mkdir $p_out_parent$k
site='UKBB11025' 

p_out=$p_out_parent$k'/fold_'$fold
echo $p_out
mkdir $p_out
    
python $hps_conf --tasks $site --type 'conn' --n_subsamp $k --conf 'AGE' --num_epochs 100 --batch_size 8 --encoder 3 --head 3 --data_format 0 --fold $fold --log_dir $p_out --data_dir $data_dir