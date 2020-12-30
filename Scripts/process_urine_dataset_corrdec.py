# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 09:34:39 2020

@author: hcji
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from rdkit import Chem
from rdkit.Chem import AllChem
from DIA.core import process_dataset, grouping_results
from DIA.iq import create_metabo_list, create_metabo_table

file_dir = 'D:/data/MTBLS816_part'
file_met = 'Data/corrdec/CorrDec.csv'
file_spectra = 'Data/corrdec/true_urine_spectra.npy'
decoy_spectra = 'Data/corrdec/decoy_urine_spectra.npy'

# true results
results, spectra = process_dataset(file_dir, file_met, file_spectra, parallel=True, energy = 30, peak_threshold=5000)
results = grouping_results(results, n_candidate=1000, rt_tol = 10)
quant_list = create_metabo_list(results, median_normalization = False, missing_value_filter = 0.3)
quant_table = create_metabo_table(quant_list, spectra, 'topN', 5)
np.save('quant_table.npy', quant_table)
np.save('quant_list.npy', quant_list)

# decoy results
decoy, decoyspectra = process_dataset(file_dir, file_met, decoy_spectra, parallel=True, energy = 30, peak_threshold=5000)
decoy = grouping_results(decoy, n_candidate=1000, rt_tol = 10)
decoy_list = create_metabo_list(decoy, median_normalization = False, missing_value_filter = 0.3)
decoy_table = create_metabo_table(decoy_list, decoyspectra, 'topN', 5)
np.save('decoy_table.npy', decoy_table)
np.save('decoy_list.npy', decoy_list)
