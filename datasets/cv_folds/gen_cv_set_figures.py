import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
gen_cv_set_figures
------------------
This script generates figures that plot the distribution of the balanced test set for each case for each confound,
as well as the same for the corresponding training set.
"""

pheno_p = '/Users/harveyaa/Documents/masters/data/pheno_26-01-22.csv'
ids_p = '/Users/harveyaa/Documents/masters/data/ids/hybrid'
out_p = './hybrid/figures'

conf = ['AGE','mean_conn','FD_scrubbed','SEX','SITE']
cases = ['SZ','ASD','BIP','DEL22q11_2','DUP22q11_2','DEL16p11_2','DUP16p11_2','DEL1q21_1','DUP1q21_1']

##############
# LOAD PHENO #
##############
pheno = pd.read_csv(pheno_p,index_col=0)

############
# LOAD IDS #
############
sel_ids = []
for case in cases:
    df = pd.read_csv(os.path.join(ids_p,f"{case}.csv"),index_col=0)
    sel_ids.append(df)
sel_ids = dict(zip(cases,sel_ids))

###############
# GEN FIGURES #
###############
for case in cases:
    all_ids = sel_ids[case]
    # PLOT TEST SET
    fig, ax = plt.subplots(len(conf),5,figsize=(20,15))
    for i,c in enumerate(conf):
            for fold in range(5):
                    ids = sel_ids[case][sel_ids[case][f"fold_{fold}"]==1].index
                    
                    sns.histplot(x=c,data=pheno[pheno.index.isin(ids)],hue=case,bins=25,ax=ax[i,fold])
                    if i == 0:
                            ax[i,fold].set_title(f'fold {fold}')
                    if c == 'SITE':
                            ax[i,fold].set_xticklabels(ax[i,fold].get_xticklabels(),rotation = 270)
    plt.suptitle(f"Balanced Test Sets - {case}\n")
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(out_p,f"{case}_test.png"),dpi=300)
    
    # PLOT TRAIN SET
    fig, ax = plt.subplots(len(conf),5,figsize=(20,15))
    for i,c in enumerate(conf):
            for fold in range(5):
                    ids = sel_ids[case][sel_ids[case][f"fold_{fold}"]==1].index
                    train_ids = set(sel_ids[case].index).difference(set(ids))
                    
                    sns.histplot(x=c,data=pheno[pheno.index.isin(train_ids)],hue=case,bins=25,ax=ax[i,fold])
                    if i == 0:
                            ax[i,fold].set_title(f'fold {fold}')
                    if c == 'SITE':
                            ax[i,fold].set_xticklabels(ax[i,fold].get_xticklabels(),rotation = 270)
    plt.suptitle(f"Training Sets - {case}\n")
    plt.tight_layout(pad=1)
    plt.savefig(os.path.join(out_p,f"{case}_train.png"),dpi=300)
