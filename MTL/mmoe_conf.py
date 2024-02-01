import os
import torch
import torch.nn as nn

from torch.utils.data import Subset
from torch.utils.data import DataLoader

from miniMTL.datasets import confDataset
from miniMTL.models import *
from miniMTL.training import Trainer
from miniMTL.mmoe import MMOEModel

from argparse import ArgumentParser

"""
hps_balanced
------------
This script is to run experiments using Hard Parameter Sharing (HPS) to predict condition status.
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tasks",help="tasks",dest='tasks',nargs='*')
    parser.add_argument("--type",help="which data type, 'conn', 'conf' or 'concat'.",default='conn')
    parser.add_argument("--encoder",help="Which encoder to use.",dest='encoder',default=3,type=int)
    parser.add_argument("--head",help="Which head to use.",dest='head',default=3,type=int)
    parser.add_argument("--num_experts",help="How many experts to use.",default=3,type=int)
    parser.add_argument("--conf",help="which confound to predict, 'SEX', 'AGE' or 'FD_scrubbed'.",default='SEX')
    parser.add_argument("--n_subsamp",help="how many subjects to subsample from UKBB sites",default=50,type=int)
    parser.add_argument("--id_dir",help="path to data ods",dest='id_dir',
                        default='/home/harveyaa/projects/def-pbellec/harveyaa/neuropsych_mtl/datasets/cv_folds/age_sex/')
    parser.add_argument("--data_dir",help="path to data dir",dest='data_dir',
                        default='/home/harveyaa/Documents/masters/data/')
    parser.add_argument("--data_format",help="data format code",dest='data_format',default=0,type=int)
    parser.add_argument("--log_dir",help="path to log_dir",dest='log_dir',default=None)
    parser.add_argument("--batch_size",help="batch size for training/test loaders",default=16,type=int)
    parser.add_argument("--lr",help="learning rate for training",default=1e-3,type=float)
    parser.add_argument("--num_epochs",help="num_epochs for training",default=100,type=int)
    parser.add_argument("--fold",help="fold of CV",default=0,type=int)
    parser.add_argument("--rand_test",help="use random test sets.",action='store_true')

    args = parser.parse_args()
    
    print('#############\n# MPS model #\n#############')
    print('Task(s): ',args.tasks)
    print('Encoder: ',args.encoder)
    print('Head(s): ',args.head)
    print('Experts: ',args.num_experts)
    print('Batch size: ',args.batch_size)
    print('LR: ', args.lr)
    print('Epochs: ', args.num_epochs)
    print('#############\n')

    # Define paths to data
    p_pheno = os.path.join(args.data_dir,'pheno_26-01-22.csv')
    p_ids = args.id_dir
    p_conn = os.path.join(args.data_dir,'connectomes')

    # Create datasets
    print('Creating datasets...')
    sites = args.tasks
    data = []
    for site in sites:
        print(site)
        data.append(confDataset(site,p_pheno,conf=args.conf,id_path=p_ids,conn_path=p_conn,n_subsamp=args.n_subsamp,type=args.type))

    print('Done!\n')
    
    # Split data & create loaders & loss fns
    loss_fns = {}
    trainloaders = {}
    testloaders = {}
    decoders = {}
    for d, site in zip(data,sites):
        train_idx, test_idx = d.split_data(random=False,fold=args.fold)
        train_d = Subset(d,train_idx)
        test_d = Subset(d,test_idx)
        trainloaders[site] = DataLoader(train_d, batch_size=args.batch_size, shuffle=True)
        testloaders[site] = DataLoader(test_d, batch_size=args.batch_size, shuffle=True)

        # Regression vs classification loss
        if args.conf in ['FD_scrubbed','AGE']:
            loss_fns[site] = nn.MSELoss()
        else:
            loss_fns[site] = nn.CrossEntropyLoss()

        decoders[site] = eval(f'head{args.head}().double()')
    
    # Create model
    model = MMOEModel(args.encoder, args.num_experts, decoders,loss_fns)
    
    # Create optimizer & trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(optimizer,log_dir=args.log_dir)

    # Train model
    trainer.fit(model,trainloaders,testloaders,num_epochs=args.num_epochs)

    # Evaluate at end
    metrics = model.score(testloaders)
    for key in metrics.keys():
        print()
        print(key)
        print('Accuracy: ', metrics[key]['accuracy'])
        print('Loss: ', metrics[key]['loss'])
        print('AUC: ', metrics[key]['auc'])
        print('F1: ', metrics[key]['f1'])
        print('Precision: ', metrics[key]['precision'])
        print('Recall: ', metrics[key]['recall'])
    print()