# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 08:15:30 2020

@author: hcji
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from DIA.core import process_dataset, grouping_results
from DIA.iq import create_metabo_list, create_metabo_table

file_dir = 'D:/data/MTBLS816_mzML'
file_met = 'HMDB/urine_metabolites.csv'
file_spectra = 'HMDB/true_urine_spectra.npy'
decoy_spectra = 'HMDB/decoy_urine_spectra.npy'

# true results
results, spectra = process_dataset(file_dir, file_met, file_spectra, energy = 30, peak_threshold=5000)
results = grouping_results(results)
quant_list = create_metabo_list(results, median_normalization = False, missing_value_filter = 0.3)
quant_table = create_metabo_table(quant_list, spectra, 'topN', 5)
np.save('quant_table.npy', quant_table)
np.save('quant_list.npy', quant_list)

# decoy results
decoy, decoyspectra = process_dataset(file_dir, file_met, decoy_spectra, energy = 30, peak_threshold=5000)
decoy = grouping_results(decoy)
decoy_list = create_metabo_list(decoy, median_normalization = False, missing_value_filter = 0.3)
decoy_table = create_metabo_table(decoy_list, decoyspectra, 'topN', 5)
np.save('decoy_table.npy', decoy_table)
np.save('decoy_list.npy', decoy_list)


quant_list = np.load('quant_list.npy', allow_pickle=True).item()

quant_table = np.load('quant_table.npy', allow_pickle=True)
decoy_table = np.load('decoy_table.npy', allow_pickle=True)

true_scores = np.array(quant_table[1])
decoy_scores = np.array(decoy_table[1])
decoy_scores[np.isnan(decoy_scores)] = 0

pvals = stats.t.sf((true_scores - np.mean(decoy_scores)) / np.std(decoy_scores), len(decoy_scores)-1)

plt.figure(dpi = 300)
plt.hist(true_scores, bins = 50, color='coral', alpha=0.7, label = 'urine')
plt.hist(decoy_scores, bins = 50, color='navy', alpha=0.7, label = 'decoy')
plt.plot([0.718, 0.718], [0, 200], color='red', label='p-val = 0.05')
plt.xlabel('p-values')
plt.ylabel('peak groups')
plt.legend()

quant_mat = quant_table[0]
sel = np.where(pvals < 0.05)[0]
sel_mat = quant_mat.iloc[sel,:]
sel_scores = true_scores[sel]

metabs = np.unique([i.split('_')[0] for i in sel_mat.index])


quant_rsd = []
for i in range(len(sel_scores)):
    x = sel_mat.iloc[i,:].values
    quant_rsd.append(np.nanstd(x) / np.nanmean(x))
    
plt.figure(dpi = 300)
plt.hist(quant_rsd, bins = 50, color='red', alpha=0.7, label = 'urine')
plt.xlabel('RSD')
plt.ylabel('peak groups')
plt.legend()

