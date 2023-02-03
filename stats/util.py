import numpy as np
import pandas as pd
import patsy as pat
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

def vec_to_connectome(a,dim=64):
    """
    Turn a vector representation of lower triangular matrix to connectome.
    
    a = vector
    dim = dimension of connectome
    
    Returns:
    dim x dim connectome
    """
    A = np.zeros((dim,dim))
    mask = np.tri(dim,dtype=bool, k=0)
    A[mask]=a
    B = np.array(A).transpose()
    np.fill_diagonal(B,0)
    return A + B

def standardize(mask,data):
    """ Standardize data. """
    scaler = StandardScaler(with_mean=False, with_std=True)
    scaler.fit(data[mask])
    standardized=scaler.transform(data)
    return standardized

def mask_cc(pheno,case,control):
    """
    pheno = df w/ all subjects
    case = col (onehot)
    control = col (onehot)
    
    Returns:
    mask = bool mask with True for subs that are case + control
    """
    mask_case = pheno[case].to_numpy(dtype=bool)
    mask_con = pheno[control].to_numpy(dtype=bool)
    return mask_case + mask_con

def case_control(pheno,case,regressors,conn,std=False):
    """
    pheno = dataframe:
        -filtered to be only relevant subjects for case control (use mask_cc)
        -case column is onehot encoded
    case = column from pheno
    regressors = list of strings, formatted for patsy
    connectomes = n_subjects x n_edges array
    
    Returns:
    table = n_edges
        - betas = the difference between case + control
        - betas_std = including standardization on controls
        - pvalues = pvalues
        - qvalues = fdr corrected pvalues alpha = 0.05
    """
    n_edges = conn.shape[1]

    betas = np.zeros(n_edges)
    betas_std = np.zeros(n_edges)
    pvalues = np.zeros(n_edges)

    formula = ' + '.join((regressors + [case]))
    dmat = pat.dmatrix(formula, pheno, return_type='dataframe',NA_action='raise')
    
    mask_std = ~pheno[case].to_numpy(dtype=bool)
    conn_std = standardize(mask_std, conn)
    
    for edge in range(n_edges):
        model = sm.OLS(conn[:,edge],dmat)
        results = model.fit()
        betas[edge] = results.params[case]
        pvalues[edge] = results.pvalues[case]
        
        if std:
            model_std = sm.OLS(conn_std[:,edge],dmat)
            results_std = model_std.fit()
            betas_std[edge] = results_std.params[case]
        
    mt = multipletests(pvalues,method='fdr_bh')
    reject = mt[0]
    qvalues = mt[1]
    
    table = pd.DataFrame(np.array([betas,betas_std,pvalues,qvalues,reject]).transpose(),
                         columns=['betas','betas_std','pvalues','qvalues','reject'])
    return table