# neuropsych_mtl
This code repository is home to my master's project.
Documentation to come.

Accompanying package: [miniMTL](https://github.com/harveyaa/miniMTL)

![image](/stats/results/effect_size_cv.png)
![image](/ML/benchmark_1/results/stratify.png)
![image](/ML/benchmark_2/results/benchmark.png)

## TODO:
- NOT URGENT:
    - reorganize & fix paths
        - neg_transfer
        - neg_transfer_rand
        - neg_transfer
            - balanced
            - random

## Directory Structure
- bash_scripts
    - SLURM submission scripts for running parts of this code 
        - Mostly for narval (compute canada) 
- conf_balancing
    - Scripts related to balancing datasets
    - general_class_balancer.py
        - General Class Balancer (GCB) implementation from mleming
    - mleming_cv_splits.py
        - Using GCB to make balanced test sets
    - mleming_datasets.py
        - Using GCB to make datasets
    - pymatch_cv_splits.py
        - Using propensity score matching with pymatch to make balanced test sets
- datasets
    - .txt files for each condition with ids for case/control to include in dataset
    - cv_folds
        - .csv files for dataset & ids in each fold for balanced test set cross validation
        - gen_cv_set_figures.py
            - Plots for the balanced test sets
    - gen_dataset_figures.py
            - Plots for the datasets 
    - figures
- ML
    - benchmark_1
        - results
        - benchmark_1.py
        - benchmark_1_plots.ipynb
    - benchmark_2
        - results
        - benchmark_2.py
        - benchmark_2_plots.ipynb
- MTL
- stats
    - results
    - dataset_exploration.ipynb
        - Plots & tables
    - generate_betamaps.py
        - script to generate output of Connectome Wide Association Studies (CWAS)
    - mtd_cv.py
        - Script for cross-validation of effect size (MTD)
    - mtd_stats.py
        - Script to generate effect sizes, p-values, confidence intervals (CIs), and count what survived FDR
    - util.py

