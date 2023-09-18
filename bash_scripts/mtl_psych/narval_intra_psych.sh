#!/bin/bash
#SBATCH --job-name=psych_mlp_intra
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=30:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/psych_intrasite.out

# PSYCH STUDY - MTL CONDITIONS - INTRASITE CV

tasks='SZ ASD BIP ADHD'

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'
id_dir='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/datasets/cv_folds/intrasite/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_paper/mtl_psych/mlp/'

# SCRIPT
hps_balanced='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/hps_balanced.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'Psych tasks MLP'
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent'fold_'$fold
    echo $p_out
    mkdir $p_out

    python $hps_balanced --tasks $tasks --type 'conn' --strategy 'balanced' --num_epochs 100 --batch_size 8 --encoder 3 --head 3 --data_format 0 --log_dir $p_out --id_dir $id_dir --data_dir $data_dir --fold $fold    
done