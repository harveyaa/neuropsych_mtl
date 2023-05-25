#!/bin/bash
#SBATCH --job-name=sex_ukbb_mtl
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=40:00:00
#SBATCH --mem=5G
#SBATCH --account=rrg-jacquese
#SBATCH --array=0-11
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/sex_ukbb_mtl_%a.out

# UKBB LEARNING CURVE STUDY - MTL SEX

K_ARRAY=(50 100 200 500 1000 1500 2000 2500 3000 3500 4000 4500)
k=${K_ARRAY[$SLURM_ARRAY_TASK_ID]}

tasks='UKBB11025 UKBB11026 UKBB11027'

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_paper16/sex/ukbb/mtl/'

# SCRIPT
hps_conf='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/hps_conf.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'UKBB sites mtl MLP on conn predicting SEX'
mkdir $p_out_parent$k

for fold in 0 1 2 3 4
do
    p_out=$p_out_parent$k'/fold_'$fold
    echo $p_out
    mkdir $p_out
   
    python $hps_conf --tasks $tasks --type 'conn' --n_subsamp $k --conf 'SEX' --num_epochs 100 --batch_size 16 --encoder 3 --head 3 --data_format 0 --log_dir $p_out --data_dir $data_dir
done