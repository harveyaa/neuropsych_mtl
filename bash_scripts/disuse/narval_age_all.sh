#!/bin/bash
#SBATCH --job-name=age_all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=20:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/age_all.out

#['ds000030', 'Cardiff', 'UKBB', 'BC', 'UCLA', 'SFARI', 'ABIDE','Orban', 'ADHD200', 'ABIDE2']
tasks='ds000030 Cardiff UKBB BC UCLA SFARI ABIDE Orban ADHD200 ABIDE2'

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/results/conf/age/all/'

# SCRIPT
hps_conf='/home/harveyaa/projects/def-pbellec/harveyaa/miniMTL/examples/hps_conf.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'All tasks MLP on conn predicting AGE'
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent'fold_'$fold
    echo $p_out
    mkdir $p_out
   
    python $hps_conf --tasks $tasks --type 'conn' --conf 'AGE' --n_subsamp 1000 --num_epochs 100 --batch_size 8 --encoder 3 --head 3 --data_format 0 --log_dir $p_out --data_dir $data_dir
done