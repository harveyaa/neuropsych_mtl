#!/bin/bash
#SBATCH --job-name=check
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=01:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/output_check.out

p_out='/home/harveyaa/projects/def-pbellec/harveyaa/data/ids/hybrid/'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

echo 'Benchmarking confound accuracy on splits...'
python benchmark2.py --p_ids $p_out --p_out $p_out