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

# decoy results
decoy, decoyspectra = process_dataset(file_dir, file_met, decoy_spectra, energy = 30, peak_threshold=5000)
decoy = grouping_results(results)
decoy_list = create_metabo_list(decoy, median_normalization = False, missing_value_filter = 0.3)
decoy_table = create_metabo_table(decoy_list, decoyspectra, 'topN', 5)
np.save('decoy_table.npy', decoy_table)

