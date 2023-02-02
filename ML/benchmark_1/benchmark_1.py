import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from argparse import ArgumentParser

####################
# DEFINE FUNCTIONS #
####################

# STRATIFY

def strat_split(pheno,case,stratify='SITE'):
    """ Creates a train/test split stratified by site. """
    strat_labels = pheno[stratify].to_numpy()

    k = 0
    # Continue doing the train/test split until there are 2 classes in the test set
    while (k==0):
        train_subs, test_subs,_,_ = train_test_split(pheno.index,
                                                    np.ones(pheno.shape[0]), # Placeholder
                                                    stratify=strat_labels)
        train_mask = pheno.index.isin(train_subs)
        test_mask = pheno.index.isin(test_subs)
        
        # Make sure there are 2 classes in the test set
        if (np.unique(pheno[test_mask][case]).shape[0] != 1):
            k=1
    return train_mask, test_mask

def strat_pred(pheno,case,ids,conn,conf_mat,clf,stratify='SITE',n=5):
    """ Predicts across stratified train/test splits.
        Reported accuracy is averaged across splits. """
    mask = pheno.index.isin(ids)
    conn_conf = np.concatenate([conn,conf_mat],axis=1)
    
    X = [conf_mat[mask],conn_conf[mask]]
    y = pheno[case].to_numpy()[mask]
    
    acc_n = np.zeros(2)
    for i in range(n):
        # Get the split masks
        train_mask,test_mask = strat_split(pheno[mask],case,stratify=stratify)
        
        # Split the data
        X_train = []
        X_test = []
        for x in X:
            X_train.append(x[train_mask])
            X_test.append(x[test_mask])
        y_train,y_test = y[train_mask],y[test_mask]
        
        # Predict
        accuracy = np.zeros(2)
        j = 0
        for x_train,x_test in zip(X_train,X_test):
            clf.fit(x_train,y_train)
            y_pred = clf.predict(x_test)
                
            accuracy[j] = accuracy_score(y_test, y_pred)
            j = j+1
        acc_n += accuracy
    acc_n = acc_n/n
    return acc_n

# LOO

def loo_split(pheno,stratify='SITE'):
    """ Creates LOO splits by stratify category. """
    strat = pheno[stratify].unique()
    splits = []
    for s in strat:
        train_subs = pheno[pheno[stratify]!=s].index
        test_subs = pheno[pheno[stratify]==s].index

        train_mask = pheno.index.isin(train_subs)
        test_mask = pheno.index.isin(test_subs)

        splits.append((train_mask,test_mask))
    return splits

def loo_pred(pheno,case,ids,conn,conf_mat,clf,stratify='SITE'):
    """ Predicts across LOO train/test splits. 
        Reported accuracy is weighted average (num in test set) across splits."""
    mask = pheno.index.isin(ids)
    conn_conf = np.concatenate([conn,conf_mat],axis=1)
    
    X = [conf_mat[mask],conn_conf[mask]]
    y = pheno[case].to_numpy()[mask]
    total_subs = y.shape[0]
    
    # Stores accuracy weighted average (num subs in test) across sites
    acc_n = np.zeros(2)
    
    # Get the splits
    splits = loo_split(pheno[mask],stratify=stratify)
    n = len(splits)
    for train_mask,test_mask in splits:
        # Split the data
        X_train = []
        X_test = []
        for x in X:
            X_train.append(x[train_mask])
            X_test.append(x[test_mask])
        y_train,y_test = y[train_mask],y[test_mask]
        test_subs = y_test.shape[0]
        
        # Predict
        accuracy = np.zeros(2)
        j = 0
        for x_train,x_test in zip(X_train,X_test):
            clf.fit(x_train,y_train)
            y_pred = clf.predict(x_test)
            
            accuracy[j] = accuracy_score(y_test, y_pred)
            j = j+1
        acc_n += test_subs*accuracy
    acc_n = acc_n/total_subs
    return acc_n

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p_pheno",help="path to pheno file",dest='p_pheno',
                        default='/Users/harveyaa/Documents/masters/data/pheno_26-01-22.csv')
    parser.add_argument("--p_conn",help="path to connectomes file",dest='p_conn',
                        default='/Users/harveyaa/Documents/masters/data/connectomes_01-12-21.csv')
    parser.add_argument("--p_ids",help="path to dataset ids",dest='p_ids')
    parser.add_argument("--p_out",help="path to outputs",dest='p_out')
    args = parser.parse_args()

    # LOAD PHENO
    pheno = pd.read_csv(args.p_pheno,index_col=0)

    # MAKE CONF MAT
    pheno_confounds = pheno[['AGE','SEX', 'FD_scrubbed', 'SITE','mean_conn']]
    pheno_confounds = pd.get_dummies(pheno_confounds,columns=['SEX','SITE'],drop_first=True)
    pheno_confounds = pheno_confounds.to_numpy()

    # LOAD IDS
    cases = ['DEL15q11_2','DUP15q11_2','DUP15q13_3_CHRNA7','DEL2q13','DUP2q13','DUP16p13_11',
            'DEL13q12_12','DUP13q12_12','DEL17p12','TAR_dup','DEL1q21_1','DUP1q21_1','DEL22q11_2',
            'DUP22q11_2','DEL16p11_2','DUP16p11_2','SZ','BIP','ASD','ADHD']
    ipc = ['SZ','BIP','ASD','ADHD']

    ids_p = '/Users/harveyaa/Documents/masters/neuropsych_mtl/datasets/{}.txt'
    ids = []
    for case in cases:
        id = pd.read_csv(ids_p.format(case),header=None)[0].to_list()
        ids.append(id)
    ids = dict(zip(cases,ids))

    # LOAD CONNECTOMES
    connectomes = pd.read_csv(args.p_conn,index_col=0).to_numpy()

    # CLASSIFIERS
    clfs = {
        'LR':LogisticRegression(class_weight='balanced',max_iter=1000),
        'SVC':SVC(C=100,class_weight='balanced',kernel='linear'),
        'Ridge':RidgeClassifier(class_weight = 'balanced'),
        'GNB':GaussianNB(),
        'RF':RandomForestClassifier(class_weight='balanced'),
        'kNN':KNeighborsClassifier(n_neighbors=1)
        }

    #################
    # STRAT BY SITE #
    #################
    print('STRATIFY BY SITE')
    cases = ['DEL15q11_2','DUP15q11_2','DUP15q13_3_CHRNA7','DEL2q13','DUP2q13','DUP16p13_11',
        'DEL13q12_12','DUP13q12_12','DEL17p12','TAR_dup','DEL1q21_1','DUP1q21_1','DEL22q11_2',
        'DUP22q11_2','DEL16p11_2','DUP16p11_2','SZ','BIP','ASD','ADHD']
    ipc = ['SZ','BIP','ASD','ADHD']

    results_master = []
    for i,case in enumerate(cases):
        print('\r {}/{}: {}                                '.format(i+1, len(cases), case))

        results = []
        for clf in clfs.values():
            res = strat_pred(pheno,case,ids[case],connectomes,pheno_confounds,clf,stratify='SITE',n=5)
            results.append(res)

        results = pd.DataFrame(results,index=clfs.keys(),columns=['conf','conf_conn'])
        results_master.append(results)

    results_master = pd.concat(results_master,keys=cases,axis=1)
    results_master.to_csv(os.path.join(args.p_out,'stratify.csv'))

    #######
    # LOO #
    #######
    print('LOO')
    loo_cases = [case for case in cases if case != 'DEL22q11_2'] # single site case

    loo_results_master = []
    for i,case in enumerate(loo_cases):
        print('\r {}/{}: {}                     '.format(i+1, len(loo_cases), case))

        results = []
        for clf in clfs.values():
            res = loo_pred(pheno,case,ids[case],connectomes,pheno_confounds,clf,stratify='SITE')
            results.append(res)

        results = pd.DataFrame(results,index=clfs.keys(),columns=['conf','conf_conn'])
        loo_results_master.append(results)

    loo_results_master = pd.concat(loo_results_master,keys=loo_cases,axis=1)
    loo_results_master.to_csv(os.path.join(args.p_out,'LOO.csv'))
