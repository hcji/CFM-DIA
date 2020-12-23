# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:28:17 2020

@author: hcji
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import nnls
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from DIA.compare import dot_similarty

indicator = IterativeImputer()

def create_metabo_list(quant_table, median_normalization = True, missing_value_filter = 0.5):
    
    if median_normalization:
        preprocessed_data = pd.DataFrame(columns = quant_table.columns)
        m, dl = [], []
        for sample, sub in quant_table.groupby('Sample'):
            dl.append(sub)
            m.append(np.nanmedian(sub['Fragment_intensity']))
        f = np.mean(m) - m
        dl_n = [dl[x]['Fragment_intensity'] + f[x] for x in range(len(m))]
        for x in range(len(dl)):
            dl[x]['Fragment_intensity'] = dl_n[x]
        preprocessed_data = pd.concat(dl)
    else:
        preprocessed_data = quant_table
    
    second_id = quant_table['Metabolite'].astype(str)
    second_id = second_id.str.cat(quant_table['Group_id'].astype(str), sep='_')
    preprocessed_data['Metabolite_Group'] = second_id
    
    samples = np.unique(preprocessed_data['Sample'])
    p_list = {}
    for prot, sub in tqdm(preprocessed_data.groupby('Metabolite_Group')):
        score = np.nansum(sub['Precursor_intensity'] * sub['Score']) / np.nansum(sub['Precursor_intensity'])
        idx = np.unique(list(sub['Fragment_mz']) + [-1])
        m = pd.DataFrame(np.full((len(idx), len(samples)), np.nan))
        m.index = idx
        m.columns = samples
        for r in sub.index:
            col = sub['Sample'][r]
            row = sub['Fragment_mz'][r]
            val = sub['Fragment_intensity'][r]
            pint = sub['Precursor_intensity'][r]
            if np.isnan(m.loc[-1, col]):
                m.loc[-1, col] = pint
            else:
                pass
            if np.isnan(m.loc[row, col]):
                m.loc[row, col] = val
            else:
                m.loc[row, col] = max(m.loc[row, col], val)
        missR = len(np.where(np.isnan( m.loc[-1,:]))[0])
        missR = missR / m.shape[1]
        
        if missR > missing_value_filter:
            continue
        else:
            p_list[prot] = dict({'quant_list': m, 'score': score})
    return p_list


def create_metabo_table(metabo_list, spectra, method = "maxLFQ", N = 3, missing_inpute = True, aggregation_in_log_space = True):
    samples = metabo_list[list(metabo_list.keys())[0]]['quant_list'].columns
    tab = pd.DataFrame(np.full((len(metabo_list), len(samples)), np.nan), columns = samples)
    tab.index = metabo_list.keys()
    annotations = []
    scores = []
    for prot, X in tqdm(metabo_list.items()):
        score = X['score']
        X = X['quant_list']
        if missing_inpute:
            try:
                X1 = indicator.fit_transform(X)
                X1 = pd.DataFrame(X1)
                X1.index = X.index
                X1.columns = X.columns
                X = X1
            except:
                pass
        scores.append(score)
        
        if method == 'maxLFQ':
            try:
                estimate, annotation = maxLFQ(X)
            except:
                estimate, annotation = topN(X, N = N, aggregation_in_log_space = aggregation_in_log_space)
        elif method == 'topN':
            estimate, annotation = topN(X, N = N, aggregation_in_log_space = aggregation_in_log_space)
        elif method == 'meanInt':
            estimate, annotation = meanInt(X, aggregation_in_log_space = aggregation_in_log_space)
        elif method == 'precursor':
            estimate, annotation = X.loc[-1,:], ''
        else:
            raise IOError ('invalid method')
        tab.loc[prot, :] = np.array(estimate)
        annotations.append(annotation)
    return tab, scores


def maxLFQ(X):
    if len(X) == 1:
        estimate = X
        annotation = 'NA'
        return estimate, annotation
    elif np.isnan(X).all(axis=None):
        estimate = None
        annotation = ''
        return estimate, annotation
    else:
        pass
    
    X = np.array(X)
    N = X.shape[1]
    cc = 0
    g = np.full(N, np.nan)
    
    def spread(i):
        g[i] = cc
        for r in range(X.shape[0]):
            if ~np.isnan(X[r, i]):
                for k in range(X.shape[1]):
                    if (~np.isnan(X[r, k])) and (np.isnan(g[k])):
                        spread(k)
    
    def maxLFQ_do(X):
        N = X.shape[1]
        AtA = np.zeros((N, N))
        Atb = np.zeros(N)
        
        for i in range(N-1):
            for j in range(i+1, N):
                r_i_j = np.nanmedian(-X[:,i] + X[:,j])               
                if np.isnan( r_i_j):
                    continue
                
                AtA[i, j] = -1
                AtA[j, i] = -1
                AtA[i, i] = AtA[i, i] + 1
                AtA[j, j] = AtA[j, j] + 1

                Atb[i] = Atb[i] - r_i_j
                Atb[j] = Atb[j] + r_i_j
        
        AA = np.hstack(( np.vstack(((2 * AtA), np.ones(N))), np.expand_dims(np.append(np.ones(N), 0), 1) ))
        bb = np.append(2 * Atb, np.nanmean(X) * N)
        
        estimate, residual = nnls(AA, bb)
        return estimate[range(N)]

    for i in range(N):
        if np.isnan(g[i]):
            cc += 1
            spread(i)
    
    w = np.full(N, np.nan)
    for i in range(1, cc+1):
        ind = np.where(g == i)[0]
        if sum(ind) == 0:
            w[ind] = np.nanmedian(X[:, ind])
        else:
            w[ind] = maxLFQ_do(X[:, ind])
    
    if np.isnan(w).all():
        estimate = w
        annotation = "NA"
    else:
        quantified_samples = np.where(~np.isnan(w))[0]
        if (g[quantified_samples] == g[quantified_samples[0]]).all():
            estimate = w
            annotation = ""
        else:
            estimate = w
            annotation = g
    return estimate, annotation
        

def topN(X, N = 3, aggregation_in_log_space = True):
    if len(X) == 1:
        estimate = X
        annotation = 'NA'
        return estimate, annotation
    
    if aggregation_in_log_space:
        v = np.nanmean(X, axis = 1)
        v_sorted = np.argsort(-v)
        out = np.nanmean(X.iloc[v_sorted[range(min(N, len(v)))],:], axis = 0)
    else:
        XX = 2 ** X
        v = np.nanmean(XX, axis = 1)
        v_sorted = np.argsort(-v)
        out = np.log2(np.nanmean(XX.iloc[v_sorted[range(min(N, len(v)))],:], axis = 0))
    estimate = out
    annotation = ''
    return estimate, annotation


def meanInt(X, aggregation_in_log_space = True):
    if len(X) == 1:
        estimate = X
        annotation = 'NA'
        return estimate, annotation    

    if aggregation_in_log_space:
        out = np.nanmean(X, axis = 0)
    else:
        out = np.log2(np.nanmean(2 ** X, axis = 0))
    estimate = out
    annotation = ''
    return estimate, annotation

