import os
import sys
import tempfile
import numpy as np
import pandas as pd
import warnings
import shutil
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns

from pymatch.Matcher import Matcher

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def save_ids(ids, path_out, case,fold=None):
    tag = f'{case}.txt' if fold is None else f'{case}_test_set_{fold}.txt'
    filename = os.path.join(path_out,tag)
    with open(filename, 'w') as file:
        for i in ids:
            file.write(f"{i}\n")

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def gen_split(dcase,dcon,yvar,test_size=0.25,seed=123):
    np.random.seed(seed)

    n_sub_test = int(dcase.shape[0]*test_size)
    case_select = np.random.choice(dcase.index,n_sub_test,replace=False)

    m = Matcher(dcase[dcase.index.isin(case_select)],dcon,yvar=yvar)
    m.fit_scores(balance=True, nmodels=100)
    m.predict_scores()
    m.match(method="min", nmatches=1, threshold=0.0009,with_replacement=False)
    return m.matched_data

def gen_unique_splits(dcase,dcon,df,yvar,test_size=0.25,min_folds=5,max_fail=100,max_acc=0.55,min_acc=0.45,min_test_size=0.15):
    clf = SVC(C=100,class_weight='balanced')
    conf = ['AGE','SEX','SITE','mean_conn','FD_scrubbed']

    accuracy = []
    subsets = []
    seeds = []
    test_sizes = []
    fail_attempts = 0
    while (len(subsets) != min_folds) and (fail_attempts < max_fail):
        seed = np.random.randint(1e5)
        try:
            with HiddenPrints():
                matches = gen_split(dcase,dcon,yvar,test_size=test_size,seed=seed)
            split_test_size = matches.shape[0]/df.shape[0]

            X = pd.get_dummies(df[conf],['SEX','SITE'])
            y = df[case]

            X_train = X[~X.index.isin(matches['og_idx'])]
            X_test = X[X.index.isin(matches['og_idx'])]
            y_train = y[~y.index.isin(matches['og_idx'])]
            y_test = y[y.index.isin(matches['og_idx'])]

            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
            acc = accuracy_score(y_test,pred)
            
            fold_ids = matches['og_idx'].to_list()
            if (min_acc < acc < max_acc) & (split_test_size > min_test_size):
                if len(subsets) == 0:
                    subsets.append(fold_ids)
                    accuracy.append(acc)
                    seeds.append(seed)
                    test_sizes.append(split_test_size)
                    print(f'Subsets: {len(subsets)}')
                # Otherwise check if duplcate with existing selections
                else:
                    duplicate = []
                    for subset in subsets:
                        diff = set(subset).difference(set(fold_ids))
                        duplicate.append(len(diff) == 0)
                        
                    if not np.any(duplicate):
                        subsets.append(fold_ids)
                        accuracy.append(acc)
                        seeds.append(seed)
                        test_sizes.append(split_test_size)
                        print(f'Subsets: {len(subsets)}')

                    else:
                        fail_attempts += 1
                        if fail_attempts % 10 == 0: print(f'Fail: {fail_attempts}')
            else:
                fail_attempts += 1
                if fail_attempts % 10 == 0: print(f'Fail: {fail_attempts}')

        except:
            fail_attempts += 1
            if fail_attempts % 10 == 0: print(f'Fail: {fail_attempts}')
            
    return subsets, seeds, accuracy, test_sizes

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p_pheno",help="path to pheno file",
                        default='/home/harveyaa/projects/def-pbellec/harveyaa/data/pheno_01-12-21.csv')
    parser.add_argument("--p_ids",help="path to dataset ids",
                        default='/home/harveyaa/projects/def-pbellec/harveyaa/data/ids/datasets')
    parser.add_argument("--p_out",help="path to outputs")
    parser.add_argument("--min_folds",help="minimum folds",type=int,default=5)
    parser.add_argument("--max_fail",help="max number of fails",type=int,default=100)
    parser.add_argument("--min_acc",help="minimum accuracy to allow",type=float,default=0.45)
    parser.add_argument("--max_acc",help="maximum accuracy to allow",type=float,default=0.55)
    parser.add_argument("--test_size",help="starting test size",type=float,default=0.3)
    parser.add_argument("--min_test_size",help="minimum allowable test size",type=float,default=0.1)
    parser.add_argument("--generate_figures",help="plots of test folds",action='store_true')
    args = parser.parse_args()

    #############
    # LOAD DATA #
    #############
    print('Loading data...')
    # pheno file
    pheno = pd.read_csv(args.p_pheno,index_col=0)

    # datasets
    cases = ['SZ','ASD','BIP','DEL22q11_2','DUP22q11_2','DEL16p11_2','DUP16p11_2','DEL1q21_1','DUP1q21_1']

    sel_ids = []
    for case in cases:
        df = pd.read_csv(os.path.join(args.p_ids,f"{case}.txt"),header=None)
        sel_ids.append(df)
    sel_ids = dict(zip(cases,sel_ids))
    print('Done!\n')

    ###################
    # GENERATE SPLITS #
    ###################
    print('Generating CV splits...')
    warnings.filterwarnings("ignore")

    all_selections = {}
    all_seeds = {}
    all_accuracy = {}
    all_test_sizes = {}
    for case in cases:
        print(case)
        control = 'CON_IPC' if case in ['SZ','ASD','BIP'] else 'non_carriers'
        conf = ['AGE','SEX','SITE','mean_conn','FD_scrubbed'] if case != 'DEL22q11_2' else ['AGE','SEX','mean_conn','FD_scrubbed']

        p = pheno[pheno.index.isin(sel_ids[case][0].to_list())]
        df_con = p[(p[control] == 1)&(p['PI'].isin(p[p[case] == 1]['PI'].unique()))][conf + [case]]
        df_case = p[p[case]==1][conf + [case]]

        df_con.loc[:,case] = df_con.loc[:,case].astype(int)
        df_case.loc[:,case] = df_case.loc[:,case].astype(int)

        selections,seeds,accuracy,test_sizes = gen_unique_splits(df_case,df_con,p,case,
                                test_size=args.test_size,
                                min_folds=args.min_folds,
                                max_fail=args.max_fail,
                                max_acc=args.max_acc,
                                min_acc=args.min_acc,
                                min_test_size=args.min_test_size)
        all_selections[case] = selections
        all_seeds[case] = seeds
        all_accuracy[case] = accuracy
        all_test_sizes[case] = test_sizes
        print()
    
    #########################
    # SAVE SUCCESFUL SPLITS #
    #########################
    temp_dir = tempfile.mkdtemp()
    for case in all_selections:
        print(case)
        print("N subsets: ", len(all_selections[case]))
        print("Mean test size: ", np.mean(all_test_sizes[case]))
        print("Mean accuracy: ", np.mean(all_accuracy[case]))

        if len(all_selections[case]) == 5:
            print('Found enough splits!')
            print(f"Saving {case}...")
            for i,ids in enumerate(all_selections[case]):
                save_ids(ids,temp_dir,case,i)
        print()

    # Clean up .txt per fold -> .csv per dataset
    tag = '{}.txt'
    tag_split = '{}_test_set_{}.txt'

    for case in all_selections:
        dataset_ids = pd.read_csv(os.path.join(args.p_ids,tag.format(case)),header=None)
        dataset_ids.set_index(0,inplace=True)

        for i in range(5):
            p_split_ids = os.path.join(temp_dir,tag_split.format(case,i))
            if os.path.exists(p_split_ids):
                s_ids = pd.read_csv(p_split_ids,header=None)
                dataset_ids[f'fold_{i}'] = 1*dataset_ids.index.isin(s_ids[0].to_list())

        dataset_ids[case] = pheno[pheno.index.isin(dataset_ids.index)][case].values.astype(int)

        if 'fold_4' in dataset_ids.columns:
            dataset_ids.to_csv(os.path.join(args.p_out,f"{case}.csv"))
    
    #####################
    # GENERATE FIGURES #
    ####################
    if args.generate_figures:
        print('Generating figures...')
        for case in all_selections:
            if len(all_selections[case]) == 5:
                all_ids = sel_ids[case]

                # PLOT TEST SET
                fig, ax = plt.subplots(len(conf),5,figsize=(15,12))
                for i,c in enumerate(conf):
                        for fold in range(5):
                                ids = pd.read_csv(os.path.join(temp_dir,f"{case}_test_set_{fold}.txt"),header=None)
                                
                                sns.histplot(x=c,data=pheno[pheno.index.isin(ids[0].to_list())],hue=case,bins=25,ax=ax[i,fold])
                                if i == 0:
                                        ax[i,fold].set_title(f'fold {fold}')
                                if fold == 0:
                                        ax[i,fold].set_xlabel('')
                                        ax[i,fold].set_ylabel(c)
                                else:
                                        ax[i,fold].set_xlabel('')
                                        ax[i,fold].set_ylabel('')
                                        ax[i,fold].set_yticklabels([])
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.1,hspace=0.2)
                plt.savefig(os.path.join(args.p_out,f"{case}_test.png"),dpi=300)

                # PLOT TRAIN SET
                fig, ax = plt.subplots(len(conf),5,figsize=(15,12))
                for i,c in enumerate(conf):
                        for fold in range(5):
                                ids_train = list(set(all_ids[0].to_list()).difference(set(ids[0].to_list())))
                                
                                sns.histplot(x=c,data=pheno[pheno.index.isin(ids_train)],hue=case,bins=25,ax=ax[i,fold])
                                if i == 0:
                                        ax[i,fold].set_title(f'fold {fold}')
                                if fold == 0:
                                        ax[i,fold].set_xlabel('')
                                        ax[i,fold].set_ylabel(c)
                                else:
                                        ax[i,fold].set_xlabel('')
                                        ax[i,fold].set_ylabel('')
                                        ax[i,fold].set_yticklabels([])
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.1,hspace=0.2)
                plt.savefig(os.path.join(args.p_out,f"{case}_train.png"),dpi=300)
            print('Done!\n')
    
    shutil.rmtree(temp_dir)
