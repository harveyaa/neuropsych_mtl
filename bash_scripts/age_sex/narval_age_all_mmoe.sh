#!/bin/bash
#SBATCH --job-name=age_all_mmoe
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=50:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/age_mmoe.out

# SIMPLE TARGETS STUDY - MMOE AGE - 36 experts

# sites with at least 30 controls
tasks='ADHD1 ADHD3 ADHD5 ADHD6 HSJ NYU SZ1 SZ2 SZ3 SZ6 Svip1 Svip2 UCLA_CB UCLA_DS1 UKBB11025 UKBB11026 UKBB11027 USM'

# DATA PATH
data_dir='/home/harveyaa/projects/def-pbellec/harveyaa/data/'

# OUT PATH
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/results_paper/age/all_mmoe/'

# SCRIPT
mmoe='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/MTL/mmoe_conf.py'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'All tasks MMOE on conn predicting AGE'
for fold in 0 1 2 3 4
do
    p_out=$p_out_parent'fold_'$fold
    echo $p_out
    mkdir $p_out

    python $mmoe --tasks $tasks --type 'conn' --n_subsamp 50 --conf 'AGE' --num_epochs 100 --batch_size 8 --encoder 3 --head 33 ---num_experts 36 -data_format 0 --fold $fold --log_dir $p_out --data_dir $data_dir
done
