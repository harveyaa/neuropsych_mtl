import pandas as pd
import numpy as np
import os 
from argparse import ArgumentParser

# SEBASTIEN URCHS
def p_permut(empirical_value, permutation_values):
    n_permutation = len(permutation_values)
    if empirical_value >= 0:
        return (np.sum(permutation_values > empirical_value)+1) / (n_permutation + 1)
    return (np.sum(permutation_values < empirical_value)+1) / (n_permutation + 1)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p_null",help="path to null models dir",
                        default='/Users/harveyaa/Documents/masters/data/null_models')
    parser.add_argument("--p_boot",help="path to bootstrap dir",
                        default='/Users/harveyaa/Documents/masters/data/bootstrap')
    parser.add_argument("--p_betas",help="path to betamaps dir",
                        default='/Users/harveyaa/Documents/masters/neuropsych_mtl/stats/results/betamaps')
    parser.add_argument("--p_out",help="path to output directory")
    args = parser.parse_args()
    
    p_null = os.path.join(args.p_null,'{}_null_model_mc.npy')
    p_boot = os.path.join(args.p_boot,'{}_bs_dist_std_mc.npy')
    p_betas = os.path.join(args.p_betas,'{}_betas.csv')
    
    all_groups = ['DEL15q11_2',
                'DUP15q11_2',
                'DUP15q13_3_CHRNA7',
                'DEL2q13',
                'DUP2q13',
                'DUP16p13_11',
                'DEL13q12_12',
                'DUP13q12_12',
                'TAR_dup',
                'DEL1q21_1',
                'DUP1q21_1',
                'DEL22q11_2',
                'DUP22q11_2',
                'DEL16p11_2',
                'DUP16p11_2',
                'SZ',
                'BIP',
                'ASD',
                'ADHD']
    
    #############
    # LOAD DATA #
    #############
    null_models = []
    betamaps = []
    betamaps_std = []
    fdr = []
    bootstrap = []

    for group in all_groups:
        null_models.append(np.load(p_null.format(group)))
        betas = pd.read_csv(p_betas.format(group))
        betamaps.append(betas['betas']) # unstd bc null models are unstd
        betamaps_std.append(betas['betas_std'])
        fdr.append(betas['reject'].sum())
        bootstrap.append(np.load(p_boot.format(group)))

    ###########
    # MTD STD #
    ###########
    print('Getting std MTD values')
    mtd_std = []
    for i,label in enumerate(all_groups):
        rank = pd.qcut(np.abs(betamaps_std[i]),10,labels=False)
        decile = []
        for k in range(betamaps_std[i].shape[0]):
            if rank[k]==9:
                decile.append(np.abs(betamaps_std[i])[k])
                
        mean_top_dec = np.mean(decile)
        mtd_std.append(mean_top_dec)
    print('Done!')
 
    ############
    # MTD PVAL #
    ############
    # Get mean top decile of null models
    mtd_null = np.zeros((len(null_models),len(null_models[0])))
    # For each null model
    for i,_ in enumerate(null_models):
        label = all_groups[i]
        print(f"Getting null model of MTD for {label}...")
        mod = null_models[i] # 5000x2080
        
        # For each iteration (5000)
        for j in range(len(null_models[0])):
            rank = pd.qcut(np.abs(mod[j,:]),10,labels=False)
            decile = []
            for k in range(mod.shape[1]):
                if rank[k]==9:
                    decile.append(np.abs(mod[j,:][k]))
            mean_top_dec = np.mean(decile)
            
            mtd_null[i,j] = mean_top_dec
        print('Done!')
    mtd_null = pd.DataFrame(np.transpose(mtd_null),columns=all_groups)
    
    print('Getting actual MTD values & calculating significance...')
    mtd = []
    p_val_mtd = []
    for i,label in enumerate(all_groups):
        rank = pd.qcut(np.abs(betamaps[i]),10,labels=False)
        decile = []
        for k in range(betamaps[i].shape[0]):
            if rank[k]==9:
                decile.append(np.abs(betamaps[i])[k])
                
        mean_top_dec = np.mean(decile)
        mtd.append(mean_top_dec)
        
        p = p_permut(mean_top_dec,mtd_null[label].values)
        p_val_mtd.append(p)
    print('Done!')

    #################
    # MTD BOOTSTRAP #
    #################
    # Get mean top decile of bootstrap models
    mtd_boot = np.zeros((len(bootstrap),len(bootstrap[0])))
    # For each bootstrap model
    for i,_ in enumerate(bootstrap):
        label = all_groups[i]
        print(f"Getting bootstrap dist of MTD for {label}...")
        mod = bootstrap[i] # 5000x2080

        # For each iteration (5000)
        for j in range(len(bootstrap[0])):
            rank = pd.qcut(np.abs(mod[j,:]),10,labels=False)
            decile = []
            for k in range(mod.shape[1]):
                if rank[k]==9:
                    decile.append(np.abs(mod[j,:][k]))
            mean_top_dec = np.mean(decile)

            mtd_boot[i,j] = mean_top_dec
        print('Done!')
    mtd_boot = pd.DataFrame(np.transpose(mtd_boot),columns=all_groups)

    ci_5 = []
    ci_95 = []
    for group in all_groups:
        ci_5.append(mtd_boot.sort_values(group)[group].to_list()[int(0.05*5000) - 1])
        ci_95.append(mtd_boot.sort_values(group)[group].to_list()[int(0.95*5000) - 1])

    mtd_pval = pd.DataFrame(np.array([mtd_std,mtd,p_val_mtd,fdr,ci_5,ci_95]).transpose(),
                            index=all_groups,columns=['mtd_std','mtd','p_permut','fdr','ci_5','ci_95'])
    mtd_pval.to_csv(os.path.join(args.p_out,'mtd_table.csv'))