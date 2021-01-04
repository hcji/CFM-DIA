# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 08:15:30 2020

@author: hcji
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from tqdm import tqdm
from scipy import stats
from rdkit import Chem
from rdkit.Chem import AllChem
from DIA.core import process_dataset, grouping_results
from DIA.iq import create_metabo_list, create_metabo_table

file_dir = 'D:/data/MTBLS816_mzML'
file_met = 'HMDB/all_metabolites.csv'
file_spectra = 'HMDB/true_urine_spectra.npy'
decoy_spectra = 'HMDB/decoy_urine_spectra.npy'

# true results
results, spectra = process_dataset(file_dir, file_met, file_spectra, parallel=True, energy = 30, peak_threshold=5000)
results = grouping_results(results, n_candidate=1000, rt_tol = 15)
quant_list = create_metabo_list(results, median_normalization = False, missing_value_filter = 0.3)
quant_table = create_metabo_table(quant_list, spectra, 'topN', 5)
np.save('quant_table.npy', quant_table)
np.save('quant_list.npy', quant_list)

# decoy results
decoy, decoyspectra = process_dataset(file_dir, file_met, decoy_spectra, parallel=True, energy = 30, peak_threshold=5000)
decoy = grouping_results(decoy, n_candidate=1000, rt_tol = 15)
decoy_list = create_metabo_list(decoy, median_normalization = False, missing_value_filter = 0.3)
decoy_table = create_metabo_table(decoy_list, decoyspectra, 'topN', 5)
np.save('decoy_table.npy', decoy_table)
np.save('decoy_list.npy', decoy_list)

n_metabolites = np.unique([i.split('_')[0] for i in quant_list.keys()])
print(len(n_metabolites))

quant_list = np.load('quant_list.npy', allow_pickle=True).item()
quant_table = np.load('quant_table.npy', allow_pickle=True)
decoy_table = np.load('decoy_table.npy', allow_pickle=True)

true_scores = np.array(quant_table[1])
decoy_scores = np.array(decoy_table[1])
decoy_scores[np.isnan(decoy_scores)] = 0

pvals = stats.t.sf((true_scores - np.mean(decoy_scores)) / np.std(decoy_scores), len(decoy_scores)-1)
thres = true_scores[np.argmin(np.abs(pvals - 0.05))]

quant_output = quant_table[0]
quant_output['MCI Score'] = quant_table[1]
quant_output['RT'] = [i.split('_')[-1] for i in quant_table[0].index]
quant_output['p value'] = pvals
quant_output.to_csv('quant_output.csv')

decoy_output = decoy_table[0]
decoy_output['MCI Score'] = decoy_table[1]
decoy_output['RT'] = [i.split('_')[-1] for i in decoy_table[0].index]
decoy_output.to_csv('decoy_output.csv')

plt.figure(dpi = 300)
plt.hist(true_scores, bins = 50, color='coral', alpha=0.7, label = 'urine')
plt.hist(decoy_scores, bins = 50, color='navy', alpha=0.7, label = 'decoy')
plt.plot([thres, thres], [0, 1200], color='red', label='p-val = 0.05')
plt.xlabel('MCI scores')
plt.ylabel('peak groups')
plt.legend()
plt.show()

quant_mat = quant_table[0]
sel1 = np.where(pvals <= 0.05)[0]
sel_mat1 = quant_mat.iloc[sel1,:]
sel2 = np.where(pvals > 0.05)[0]
sel_mat2 = quant_mat.iloc[sel2,:]
n_met = np.unique([i.split('_')[0] for i in sel_mat1.index])

corr_metab = pd.read_csv('Data/corrdec/CorrDec.csv')
common = []
for i in sel_mat1.index:
    n = i.split('_')[0]
    rt = float(i.split('_')[-1])
    wh = np.where(corr_metab['Metabolite'] == n)[0]
    if len(wh) == 0:
        continue
    else:
        wh = wh[0]
        rt1 = corr_metab['Rt'][wh]
        if abs(rt - rt1) < 20:
            if n not in common:
                common.append(n)
common = np.unique(common)
a = len(common)
b = len(n_met) - a
c = len(corr_metab) - a
plt.figure(dpi = 300)
venn2((b, c, a), set_labels = ('CFM-DIA (p < 0.05)', 'CorrDec'))


high_score = np.where(pvals < 0.001)[0]
tmp = quant_list[list(quant_list.keys())[8627]]


quant_rsd1 = []
for i in range(len(sel1)):
    x = sel_mat1.iloc[i,:].values
    quant_rsd1.append(np.nanstd(x) / np.nanmean(x))
quant_rsd2 = []
for i in range(len(sel2)):
    x = sel_mat2.iloc[i,:].values
    quant_rsd2.append(np.nanstd(x) / np.nanmean(x))
    
plt.figure(dpi = 300)
plt.hist(quant_rsd1, bins = 50, color='red', alpha=0.5, label = 'p-val < 0.05')
plt.hist(quant_rsd2, bins = 50, color='navy', alpha=0.5, label = 'p-val > 0.05')
plt.xlabel('RSD')
plt.ylabel('peak groups')
plt.legend()
plt.show()


k1 = 'Glutamine_51385_533.14'
k2 = 'N-acetylcarnosine_69554_534.6'
ms1 = np.array([list(quant_list[k1]['quant_list'].index[1:]), quant_list[k1]['quant_list'].iloc[1:,0].values]).T
ms2 = np.array([list(quant_list[k2]['quant_list'].index[1:]), quant_list[k2]['quant_list'].iloc[1:,0].values]).T

ms1_ref = pd.read_csv('Glutamine.csv', header=None)
ms2_ref = pd.read_csv('N-Acetylcarnosine.csv', header=None)

def plot_compare_ms(spectrum1, spectrum2, tol=0.05):
    spectrum1['intensity'] /= max(spectrum1['intensity'])
    spectrum2['intensity'] /= max(spectrum2['intensity'])
    
    c_mz = []
    c_int = []
    for i in spectrum1.index:
        diffs = abs(spectrum2['mz'] - spectrum1['mz'][i])
        if min(diffs) < tol:
            c_mz.append(spectrum1['mz'][i])
            c_mz.append(spectrum2['mz'][np.argmin(diffs)])
            c_int.append(spectrum1['intensity'][i])
            c_int.append(-spectrum2['intensity'][np.argmin(diffs)])
    c_spec = pd.DataFrame({'mz':c_mz, 'intensity':c_int}) 
    
    plt.figure(dpi = 300)
    plt.vlines(spectrum1['mz'], np.zeros(spectrum1.shape[0]), np.array(spectrum1['intensity']), 'gray')
    plt.axhline(0, color='black')
    plt.vlines(spectrum2['mz'], np.zeros(spectrum2.shape[0]), -np.array(spectrum2['intensity']), 'gray')
    plt.vlines(c_spec['mz'], np.zeros(c_spec.shape[0]), c_spec['intensity'], 'red')
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')
    plt.show()

plot_compare_ms(ms1, ms1_ref)
plot_compare_ms(ms2, ms2_ref)