import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
gen_dataset_figures
-------------------
This script generates figures that plot the distribution of the dataset for each case for each confound.
Both the entire dataset (all cases & all controls from each site with a control from the given condition),
and the selected subset (all cases and closely matched controls) are plotted.
"""

pheno_p = '/Users/harveyaa/Documents/masters/data/pheno_26-01-22.csv'
out_p = './figures'

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
    df = pd.read_csv(f"{case}.txt",header=None)
    sel_ids.append(df)
sel_ids = dict(zip(cases,sel_ids))

###############
# GEN FIGURES #
###############
for case in cases:
    ids = sel_ids[case]
    
    control = 'CON_IPC' if case in ['ASD','SZ','BIP'] else 'non_carriers'
    
    sites = pheno[pheno[case]==1]['SITE'].unique()
    df = pheno[(pheno['SITE'].isin(sites))
            & ((pheno[case]==1)|(pheno[control]==1))]

    fig, ax = plt.subplots(len(conf),2,figsize=(9,12))

    for i,c in enumerate(conf):
        sns.histplot(x=c,data=df,hue=case,bins=25,ax=ax[i,0])
            
        sns.histplot(x=c,data=df[df.index.isin(ids[0].to_list())],hue=case,bins=25,ax=ax[i,1])
        if i == 0:
                ax[i,0].set_title('Full dataset',fontsize=13)
                ax[i,1].set_title('Subset',fontsize=13)
        
        if c == 'SITE':
                ax[i,0].set_xticklabels(ax[i,0].get_xticklabels(),rotation = 270)
                ax[i,1].set_xticklabels(ax[i,1].get_xticklabels(),rotation = 270)
    fig.suptitle(case)
    plt.tight_layout(pad = 1.5)
    plt.savefig(os.path.join(out_p,f"{case}.png"),dpi=300)