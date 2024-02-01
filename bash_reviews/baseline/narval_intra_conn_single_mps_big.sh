#!/bin/bash
#SBATCH --job-name=single_conn_intrasite
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=5:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH --array=9-10
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/conn_intrasite_%a.out

# CNVs & PSYCH STUDY - SINGLE TASK CONDITIONS - INTRASITE CV
# NEGATIVE TRANSFER STUDY - mps big BASELINE

TASK_ARRAY=('SZ' 'ASD' 'BIP' 'DEL22q11_2' 'DUP22q11_2' 'DEL16p11_2' 'DUP16p11_2' 'DEL1q21_1' 'DUP1q21_1' 'ADHD' 'DEL15q11_2')
task=${TASK_ARRAY[$SLURM_ARRAY_TASK_ID]}

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'
id_dir='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/datasets/cv_folds/intrasite/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_reviews/baseline/mps_big/'

# SCRIPT
mps='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/mps.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env2/bin/activate

echo 'Single task mps big on conn w/ intra-site CV on '$task
mkdir $p_out_parent$task
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent$task'/fold_'$fold
    echo $p_out
    mkdir $p_out

    python $mps --tasks $task --type 'conn' --num_epochs 100 --batch_size 8 --preencoder 333 --encoder 333 --head 3 --data_format 0 --log_dir $p_out --id_dir $id_dir --data_dir $data_dir --fold $fold
done