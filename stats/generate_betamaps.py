import os
import util
import numpy as np
import pandas as pd
from argparse import ArgumentParser

def get_year(s):
    if isinstance(s,str):
        return int(s.split('/')[-1])
    else:
        return np.nan

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p_pheno",help="path to pheno file",dest='p_pheno',
                        default='/Users/harveyaa/Documents/masters/data/pheno_26-01-22.csv')
    parser.add_argument("--p_conn",help="path to connectomes file",dest='p_conn',
                        default='/Users/harveyaa/Documents/masters/data/connectomes_01-12-21.csv')
    parser.add_argument("--p_out",help="path to outputs",dest='p_out')
    args = parser.parse_args()
 
    #############
    # LOAD DATA #
    #############
    pheno = pd.read_csv(args.p_pheno,index_col=0)
    connectomes = pd.read_csv(args.p_conn,index_col=0)

    conf = ['AGE','C(SEX)','FD_scrubbed', 'C(SITE)', 'mean_conn']

    ############
    # BETAMAPS #
    ############

    cases =['SZ','BIP','ASD','ADHD','DEL1q21_1','DEL2q13','DEL13q12_12','DEL15q11_2','DEL16p11_2',
            'DEL22q11_2','TAR_dup','DUP1q21_1','DUP2q13','DUP13q12_12','DUP15q11_2','DUP15q13_3_CHRNA7',
            'DUP16p11_2','DUP16p13_11','DUP22q11_2']
    ipc = ['SZ','BIP','ASD','ADHD']
    
    df_pi = pheno.groupby('PI').sum()[cases]
    mask_pi = (df_pi > 0)

    # MEAN CORRECTED
    summaries = []
    for case in cases:
        if case in ipc:
            mask = util.mask_cc(pheno,case,'CON_IPC')
        else:
            mask_case = pheno[case].to_numpy(dtype=bool)
            pi_list = df_pi[mask_pi[case]].index.to_list()
            mask_con = np.array((pheno['PI'].isin(pi_list))&(pheno['non_carriers']==1))
            mask = mask_case + mask_con
            print(case,pi_list)

        summary = util.case_control(pheno[mask],case,conf,connectomes.to_numpy()[mask],std=True)
        summary.to_csv(args.p_out + '/{}_betas.csv'.format(case))
        
        summaries.append(summary)
        print('Completed {}.'.format(case))