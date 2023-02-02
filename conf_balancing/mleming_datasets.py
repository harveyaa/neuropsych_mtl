import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from argparse import ArgumentParser

from general_class_balancer import *

"""
GENERATE DATASETS
-----------------
This script is to generate datasets (sets of subject ids) that are as balanced as possible for binary 
classification wrt confounding factors from the entire pool of controls.
- Idiopathic conditions (IPC) - use all controls from the same PI.
- Most CNVs - use Leming's class balancing algorithm.
- DUP16p11_2 - use Leming's class balancing algorithm for 34/35 subjects, 
    add last subject & hand select final control. 
- DEL22q11_2 - special case, take all controls from the site.
"""

def save_ids(ids, path_out, case):
    tag = f'{case}.txt'
    filename = os.path.join(path_out,tag)
    with open(filename, 'w') as file:
        for i in ids:
            file.write(f"{i}\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p_pheno",help="path to pheno file",dest='p_pheno',
                        default='/Users/harveyaa/Documents/masters/data/pheno_26-01-22.csv')
    parser.add_argument("--p_out",help="path to outputs")
    parser.add_argument("--generate_figures",help="plots of datasets",action='store_true')
    args = parser.parse_args()

    #############
    # LOAD DATA #
    #############
    print('Loading data...')
    pheno = pd.read_csv(args.p_pheno,index_col=0)
    print('Done!\n')

    conf = ['AGE','SEX','SITE','mean_conn','FD_scrubbed']

    print('Generating datasets...\n')

    #######
    # IPC #
    #######
    ipc = ['SZ','ASD','BIP','ADHD']
    for case in ipc:  
        print(case)
        control = 'CON_IPC'

        # Take all controls from PIs w/ case
        df_con = pheno[(pheno[control] == 1)&(pheno['PI'].isin(pheno[pheno[case] == 1]['PI'].unique()))]
        df = pd.concat([df_con,pheno[pheno[case]==1]])
        df.loc[:,case] = df.loc[:,case].astype(int)

        # Save those ids
        print('Saving selection...\n')
        save_ids(df.index.to_list(),args.p_out,case)
    
    #############
    # MOST CNVs #
    #############
    cnvs = [#'DEL22q11_2',
            'DUP22q11_2',
            'DEL16p11_2',
            #'DUP16p11_2',
            'DEL1q21_1',
            'DUP1q21_1',
            'DEL15q11_2',
            'DUP15q11_2',
            'DUP15q13_3_CHRNA7',
            'DEL2q13',
            'DUP2q13',
            'DUP16p13_11',
            'DEL13q12_12',
            'DUP13q12_12',
            'DEL17p12',
            'TAR_dup']
    
    for case in cnvs:
        print(case)
        control = 'non_carriers'

        # Take all controls from PIs w/ case
        df_con = pheno[(pheno[control] == 1)&(pheno['PI'].isin(pheno[pheno[case] == 1]['PI'].unique()))]
        df = pd.concat([df_con,pheno[pheno[case]==1]])
        df.loc[:,case] = df.loc[:,case].astype(int)

        # Make confound array
        confounds = df[conf].transpose().values
        labels = df[case].values.astype(int)

        print('Finding balanced matches...')
        n_case = np.sum(labels)
        print('Total cases: ', n_case)
        selected_case = 0
        while selected_case != n_case:
            selection = class_balance(labels,confounds)
            selected_case = np.sum(labels[selection])
            print('Selected cases: ',selected_case)

        print('Saving selection...\n')
        save_ids(df[selection].index.to_list(),args.p_out,case)
    
    ##############
    # DUP16p11_2 #
    ##############
    print('DUP16p11_2')

    # Take all controls from PIs w/ case
    df_con = pheno[(pheno['non_carriers'] == 1)&(pheno['PI'].isin(pheno[pheno['DUP16p11_2'] == 1]['PI'].unique()))]
    df = pd.concat([df_con,pheno[pheno['DUP16p11_2']==1]])
    df.loc[:,'DUP16p11_2'] = df.loc[:,'DUP16p11_2'].astype(int)

    # Make confound array
    confounds = df[conf].transpose().values
    labels = df['DUP16p11_2'].values.astype(int)

    print('Finding balanced matches...')
    n_case = np.sum(labels)
    print('Total cases: ', n_case)
    selected_case = 0
    dup16p_ids = []
    while selected_case != n_case:
        selection = class_balance(labels,confounds)
        selected_case = np.sum(labels[selection])
        print('Selected cases: ',selected_case)

        if selected_case == 34:
            print('Found close solution...')
            dup16p_ids.append(df[selection].index.to_list())
        
        if len(dup16p_ids) == 5:
            print()
            break
    
    # Check that excluded case is always the same
    excluded = []
    for i in range(5):
        all_dup16 = pheno[pheno['DUP16p11_2']==1].index
        sel_dup16 = pheno[(pheno.index.isin(dup16p_ids[i])) & (pheno['DUP16p11_2']==1)].index
        excluded_case = list(set(all_dup16).difference(set(sel_dup16)))[0]
        excluded.append(excluded_case)
    assert len(np.unique(excluded)) == 1
    print('Same subjected excluded from each close solution: ',excluded[0])

    # Check that the controls are always the same
    diff_controls = []
    for i,j in combinations(range(5),2):
        sel_con_i = pheno[(pheno.index.isin(dup16p_ids[i])) & (pheno['DUP16p11_2']==0)].index
        sel_con_j = pheno[(pheno.index.isin(dup16p_ids[i])) & (pheno['DUP16p11_2']==0)].index
        diff = len(set(sel_con_i).difference(set(sel_con_j)))
        diff_controls.append(diff)
    assert np.sum(diff_controls) == 0
    print('Same control subjects in each close solution.')

    print('Excluded case: ', pheno.loc[excluded[0]][conf])

    # Handpick youngest 'matched' control not already in selection
    handpick_con = pheno[(pheno['SITE'] == 'Svip1') 
                    & (pheno['SEX'] == 'Male')
                    & (pheno['non_carriers'] == 1)
                    & (~pheno.index.isin(dup16p_ids[0]))][conf].sort_values('AGE').index[0]

    print('Handpicked control: ', pheno.loc[handpick_con][conf])

    dup16p_hand_selection = dup16p_ids[0] + [excluded[0]] + [handpick_con]

    print('Saving selection...\n')
    save_ids(dup16p_hand_selection,args.p_out,'DUP16p11_2')

    ##############
    # DEL22q11_2 #
    ##############
    print('DEL22q11_2')

    # Take all the cases and controls from the site
    del22q_con_idx = pheno[(pheno['SITE']=='UCLA_CB') & (pheno['non_carriers']==1)].index.to_list()
    del22q_case_idx = pheno[(pheno['SITE']=='UCLA_CB') & (pheno['DEL22q11_2']==1)].index.to_list()

    print('Saving selection...\n')
    save_ids(del22q_con_idx + del22q_case_idx,args.p_out,'DEL22q11_2')

    ####################
    # GENERATE FIGURES #
    ####################
    if args.generate_figures:
        print('Generating figures...')

        cases = ['SZ','ASD','BIP','DEL22q11_2','DUP22q11_2','DEL16p11_2','DUP16p11_2','DEL1q21_1','DUP1q21_1']

        sel_ids = []
        for case in cases:
            df = pd.read_csv(os.path.join(args.p_out,f"{case}.txt"),header=None)
            sel_ids.append(df)
        sel_ids = dict(zip(cases,sel_ids))

        for case in cases:
            ids = sel_ids[case]
            
            control = 'CON_IPC' if case in ['ASD','SZ','BIP'] else 'non_carriers'
            
            sites = pheno[pheno[case]==1]['SITE'].unique()
            df = pheno[(pheno['SITE'].isin(sites))
                    & ((pheno[case]==1)|(pheno[control]==1))]

            fig, ax = plt.subplots(len(conf),2,figsize=(7,12))

            for i,c in enumerate(conf):
                    sns.histplot(x=c,data=df,hue=case,bins=25,ax=ax[i,0])
                    
                    sns.histplot(x=c,data=df[df.index.isin(ids[0].to_list())],hue=case,bins=25,ax=ax[i,1])

                    if i == 0:
                            ax[i,0].set_title('Full dataset')
                            ax[i,1].set_title('Subset')

            plt.tight_layout()
            plt.savefig(os.path.join(args.p_out,f"{case}.png"),dpi=300)
        print('Done!\n')
