# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 08:32:07 2020

@author: hcji
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from DIA.core import process_dataset, grouping_results
from DIA.iq import create_metabo_list, create_metabo_table

concent = np.array([4000, 2700, 1800, 1200, 790, 530, 350, 230, 160, 100, 69])

all_dirs = os.listdir('D:/data/MTBLS787_mzML')
for file_dir in all_dirs:
    metabolite = file_dir.split('.')[0]
    file_met = 'Data/benchmarks/{}.csv'.format(metabolite)
    file_dir = 'D:/data/MTBLS787_mzML/' + file_dir

    results, spectra = process_dataset(file_dir, file_met, energy = 30, peak_threshold=3000)
    results = grouping_results(results)

    quant_list = create_metabo_list(results, median_normalization = False, missing_value_filter = 0.1)
    quant_table = create_metabo_table(quant_list, spectra, method = "topN", N = 5)
    
    # decoy results
    decoy_results, decoy_spectra = process_dataset(file_dir, file_met, energy = 30, peak_threshold=3000, use_decoy=True)
    decoy_results = grouping_results(decoy_results)

    decoy_list = create_metabo_list(decoy_results, median_normalization = False, missing_value_filter = 0.1)
    decoy_table = create_metabo_table(decoy_list, spectra, method = "topN", N = 5)
    
    s = np.array([int(re.findall(r'\d+', s.split('_')[-2])[0]) - 2  for s in quant_table[0].columns])
    wh = np.argmax(np.nansum(quant_table[0], 1))
    
    corr = round(np.corrcoef(np.log2(concent[s]), np.log2(quant_table[0].iloc[wh,:]))[0,1], 3)

