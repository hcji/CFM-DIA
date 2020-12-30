# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 09:19:15 2020

@author: hcji
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
from PyCFMID.PyCFMID import cfm_predict
from DIA.utils import get_decoy_spectrum

corrdec_met = pd.read_csv('Data/corrdec/CorrDec.csv')
smiles = corrdec_met['SMILES']

spectra = []
for smi in tqdm(smiles):
    try:
        spectrum = cfm_predict(smi, prob_thresh=0.01, param_file='', config_file='', annotate_fragments=False, output_file=None, apply_postproc=True, suppress_exceptions=False)
    except:
        spectrum = None
    spectra.append(spectrum)
np.save('Data/corrdec/corrdec_spectra.npy', spectra)

spectra = np.load('Data/corrdec/corrdec_spectra.npy', allow_pickle=True)
new_metab = pd.DataFrame(columns = corrdec_met.columns)
true_spectra, decoy_spectra = dict(), dict()
for i in tqdm(range(len(spectra))):
    metabolite = corrdec_met.loc[i, 'Metabolite']
    precursor_mz = corrdec_met.loc[i, 'Precursor_mz']
    
    if metabolite in list(new_metab['Metabolite']):
        continue
    else:
        new_metab.loc[len(new_metab)] = corrdec_met.loc[i, :]
    
    if np.isnan(precursor_mz):
        true_spectra[metabolite] = spectra[i]['medium_energy']
        decoy_spectra[metabolite] = None
    else:
        decoy_spectrum = get_decoy_spectrum(precursor_mz, spectra[i]['medium_energy'])
        true_spectra[metabolite] = spectra[i]['high_energy']
        decoy_spectra[metabolite] = decoy_spectrum

np.save('Data/corrdec/true_urine_spectra.npy', true_spectra)
np.save('Data/corrdec/decoy_urine_spectra.npy', decoy_spectra)
new_metab.to_csv('Data/corrdec/CorrDec.csv', index = False)