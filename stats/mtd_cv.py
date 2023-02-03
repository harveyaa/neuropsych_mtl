import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser

import random
import patsy as pat
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

def standardize(mask,data):
    scaler = StandardScaler(with_mean=False, with_std=True)
    scaler.fit(data[mask])
    standardized=scaler.transform(data)
    return standardized

# redefined function from util to only generate standardized betamaps
def case_control(pheno,case,regressors,conn):
    """
    pheno = dataframe:
        -filtered to be only relevant subjects for case control (use mask_cc)
        -case column is onehot encoded
    case = column from pheno
    regressors = list of strings, formatted for patsy
    connectomes = n_subjects x n_edges array
    
    Returns:
    table = n_edges
        - betas_std = including standardization on controls
        - pvalues = pvalues
        - qvalues = fdr corrected pvalues alpha = 0.05
    """
    n_edges = conn.shape[1]

    betas_std = np.zeros(n_edges)

    formula = ' + '.join((regressors + [case]))
    dmat = pat.dmatrix(formula, pheno, return_type='dataframe',NA_action='raise')
    
    mask_std = ~pheno[case].to_numpy(dtype=bool)
    conn_std = standardize(mask_std, conn)
    
    for edge in range(n_edges):
        model_std = sm.OLS(conn_std[:,edge],dmat)
        results_std = model_std.fit()
        betas_std[edge] = results_std.params[case]
    
    table = pd.DataFrame(np.array([betas_std]).transpose(),
                         columns=['betas_std'])
    return table

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p_pheno",help="path to pheno file",
                        default='/Users/harveyaa/Documents/masters/data/pheno_26-01-22.csv')
    parser.add_argument("--p_conn",help="path to connectomes file",
                        default='/Users/harveyaa/Documents/masters/data/connectomes_01-12-21.csv')
    parser.add_argument("--p_out",help="path to output directory")
    parser.add_argument("--n_folds",help="Number of folds of cross_validation",type=int,default=5)
    args = parser.parse_args()
    
    #############
    # LOAD DATA #
    #############
    
    print('Loading data...')
    pheno = pd.read_csv(args.p_pheno,index_col=0)
    connectomes = pd.read_csv(args.p_conn,index_col=0)
    print('Done!')

    regressors = ['AGE','C(SEX)','FD_scrubbed', 'C(SITE)', 'mean_conn']

    ################
    # CASE CONTROL #
    ################

    cases =['SZ',
            'BIP',
            'ASD',
            'ADHD',
            'DEL1q21_1',
            'DEL2q13',
            'DEL13q12_12',
            'DEL15q11_2',
            'DEL16p11_2',
            'DEL22q11_2',
            'TAR_dup',
            'DUP1q21_1',
            'DUP2q13',
            'DUP13q12_12',
            'DUP15q11_2',
            'DUP15q13_3_CHRNA7',
            'DUP16p11_2',
            'DUP16p13_11',
            'DUP22q11_2']
    ipc = ['SZ','BIP','ASD','ADHD']
    
    df_pi = pheno.groupby('PI').sum()[cases]
    mask_pi = (df_pi > 0)
    
    cc_rows = []
    for case in cases:
        print(f'{case} - estimating effect size...')
        # SELECT SUBJECTS
        if case in ipc:
            mask_case = pheno[case].to_numpy(dtype=bool)
            mask_con = pheno['CON_IPC'].to_numpy(dtype=bool)
            mask = mask_case + mask_con
        elif case == 'IBD':
            mask_case = (pheno['IBD_str'] == 'IBD_K50_K51').to_numpy(dtype=bool)
            mask_con = (pheno['IBD_str'] == 'no_IBD').to_numpy(dtype=bool)
            mask = mask_case + mask_con
        else:
            mask_case = pheno[case].to_numpy(dtype=bool)
            pi_list = df_pi[mask_pi[case]].index.to_list()
            mask_con = np.array((pheno['PI'].isin(pi_list))&(pheno['non_carriers']==1))
            mask = mask_case + mask_con
            print(case,pi_list)
        
        idx = pheno[mask].index
        strat_col = pheno[mask][case]
        
        skf = StratifiedKFold(n_splits=args.n_folds)
        
        es_train = []
        es_test = []
        split_train_idx = []
        split_test_idx = []
        k=0
        for train_index, test_index in skf.split(idx, strat_col):
            print(f'Fold {k+1}/{args.n_folds}...')
            # Save split indexes for the bootstrap
            split_train_idx.append(idx[train_index].to_list())
            split_train_idx.append(idx[test_index].to_list())
            
            betamap_train = case_control(pheno.loc[idx[train_index]],
                                        case,
                                        regressors,
                                        connectomes.loc[idx[train_index]].to_numpy())['betas_std']

            rank = pd.qcut(betamap_train.abs(),10,labels=False)
            decile_idx = rank[rank==9].index
            
            # Get train ES
            decile_train = betamap_train.abs()[decile_idx]
            mtd_train = np.mean(decile_train)
            es_train.append(mtd_train)

            betamap_test = case_control(pheno.loc[idx[test_index]],
                                        case,
                                        regressors,
                                        connectomes.loc[idx[test_index]].to_numpy())['betas_std']
            # Get test ES
            decile_test = betamap_test.abs()[decile_idx]
            mtd_test = np.mean(decile_test)
            es_test.append(mtd_test)
            
            k += 1
            
        print(f'{case} - CV Effect size ',np.mean(es_test))
        cc_rows.append([case,np.mean(es_train),np.mean(es_test)])
        df_split_idx = pd.DataFrame(zip(split_train_idx,split_train_idx),columns=['train','test'])
        
    print('Saving results...')
    df_cc = pd.DataFrame(cc_rows,columns=['case','train_ES','test_ES'])
    df_cc.to_csv(os.path.join(args.p_out,f'mtd_CV_{args.n_folds}.csv'))
    print('Done!')