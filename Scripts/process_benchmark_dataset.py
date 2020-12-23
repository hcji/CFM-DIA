# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 08:32:07 2020

@author: hcji
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DIA.core import process_dataset, grouping_results
from DIA.iq import create_metabo_list, create_metabo_table
from DIA.compare import dot_similarty

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


concent = np.array([4000, 2700, 1800, 1200, 790, 530, 350, 230, 160, 100, 69])

all_dirs = os.listdir('D:/data/MTBLS787_mzML')
for file_dir in all_dirs:
    metabolite = file_dir.split('.')[0]
    file_met = 'Data/benchmarks/{}.csv'.format(metabolite)
    file_dir = 'D:/data/MTBLS787_mzML/' + file_dir

    results, spectra = process_dataset(file_dir, file_met, energy = 30, peak_threshold=3000)
    results = grouping_results(results, rt_tol = 5, n_candidate = 100)

    quant_list = create_metabo_list(results, median_normalization = False, missing_value_filter = 0.1)
    quant_table = create_metabo_table(quant_list, spectra, method = "topN", N = 5)
    quant_table1 = create_metabo_table(quant_list, spectra, method = "precursor", N = 5)
    quant_table2 = create_metabo_table(quant_list, spectra, method = "maxLFQ", N = 5)
    
    # decoy results
    decoy_results, decoy_spectra = process_dataset(file_dir, file_met, energy = 30, peak_threshold=3000, use_decoy=True)
    decoy_results = grouping_results(decoy_results, rt_tol = 5, n_candidate = 100)

    decoy_list = create_metabo_list(decoy_results, median_normalization = False, missing_value_filter = 0.1)
    decoy_table = create_metabo_table(decoy_list, spectra, method = "topN", N = 5)
    
    s = np.array([int(re.findall(r'\d+', s.split('_')[-2])[0]) - 2  for s in quant_table[0].columns])
    wh = np.argmax(np.nansum(quant_table[0], 1))
    
    corr = round(np.corrcoef(np.log2(concent[s]), np.log2(quant_table[0].iloc[wh,:]))[0,1], 3)
    corr1 = round(np.corrcoef(np.log2(concent[s]), np.log2(quant_table1[0].iloc[wh,:]))[0,1], 3)
    corr2 = round(np.corrcoef(np.log2(concent[s]), np.log2(quant_table2[0].iloc[wh,:]))[0,1], 3)

    met_group = quant_table[0].index[wh]
    extract_ms = results[np.array(results['Metabolite_Group']) == met_group]
    extract_ms = extract_ms[np.array(extract_ms['Precursor_intensity']) == max(extract_ms['Precursor_intensity'])]
    extract_ms = extract_ms.loc[:, ['Fragment_mz', 'Fragment_intensity']]
    
    true_ms = spectra[metabolite]
    extract_ms.columns = true_ms.columns
    
    plot_compare_ms(extract_ms, true_ms)
    
    correlation = []
    correlation += [-0.43012423,  0.92899437,  0.98127264, -0.41567681,  0.99924506, 0.99860792,  0.98004232,  0.99996284]
    plt.figure(dpi = 300)
    m = decoy_list[met_group]['quant_list']
    for i in m.index:
        plt.plot(np.log2(concent[s]), np.log2(np.array(m.loc[i,:])), color = 'gray', marker = 'o', alpha = 0.1)
    m = quant_list[met_group]['quant_list']
    for i in m.index:
        plt.plot(np.log2(concent[s]), np.log2(np.array(m.loc[i,:])), color = 'red', marker = 'o')
    plt.xlabel('log2(concentration)')
    plt.ylabel('log2(ion intensity)')
    plt.show()