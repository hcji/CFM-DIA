# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 07:31:23 2020

@author: hcji
"""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from molmass import Formula
from rdkit import Chem
from rdkit.Chem import AllChem
from PyCFMID.PyCFMID import cfm_predict
from DIA.utils import get_decoy_spectrum

metabolite_json = 'HMDB/urine_metabolites.json'

with open(metabolite_json, 'r') as read_file:
    metabo_data = json.load(read_file)

hmdb, metab, smiles, precursors = [], [], [], []
for item in metabo_data:
    # combine isomers as their spectra are very similar
    try:
        precursor = Formula(item['chemical_formula']).isotope.mass + 1.0078
    except:
        precursor = np.nan
    
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(item['smiles']), isomericSmiles=False)
    except:
        smi = item['smiles']
    
    if smi not in smiles:
        precursors.append(precursor)
        metab.append(item['name'])
        hmdb.append(item['accession'])
        smiles.append(item['smiles'])

hmdb_metab = pd.DataFrame({'ID': hmdb, 'Metabolite': metab, 'Adduct': '[M+H]+', 'Precursor_mz': precursors, 'SMILES': smiles})
# hmdb_metab.to_csv('HMDB/urine_metabolites.csv', index = False)

spectra = []
for smi in tqdm(smiles):
    try:
        spectrum = cfm_predict(smi, param_file='', config_file='', annotate_fragments=False, output_file=None, apply_postproc=True, suppress_exceptions=False)
    except:
        spectrum = None
    spectra.append(spectrum)
np.save('HMDB/urine_spectra.npy', spectra)

corr_spectra = []
corr_index, corr_metab, corr_smiles, corr_precursors = [], [], [], []
corrdec_met = pd.read_csv('Data/corrdec/CorrDec.csv')
for i in tqdm(range(len(corrdec_met))):
    smi = corrdec_met['SMILES'][i]
    try:
        m = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(m, isomericSmiles=False)
        f = AllChem.CalcMolFormula(m)
    except:
        continue
        
    try:
        spectrum = cfm_predict(smi, param_file='', config_file='', annotate_fragments=False, output_file=None, apply_postproc=True, suppress_exceptions=False)
    except:
        spectrum = None
    
    name = corrdec_met['Metabolite'][i]
    precursor = Formula(f).isotope.mass + 1.0078
    index = 'CorrDec_{}'.format(i)
    
    if smi not in smiles:
        corr_precursors.append(precursor)
        corr_metab.append(name)
        corr_index.append(index)
        corr_smiles.append(smi)
        corr_spectra.append(spectrum)
corr_metab = pd.DataFrame({'ID': corr_index, 'Metabolite': corr_metab, 'Adduct': '[M+H]+', 'Precursor_mz': corr_precursors, 'SMILES': corr_smiles})
corr_metab.to_csv('HMDB/corr_metabolites.csv', index = False)
np.save('HMDB/corr_spectra.npy', corr_spectra)


spectra = list(np.load('HMDB/urine_spectra.npy', allow_pickle=True))
corr_spectra = list(np.load('HMDB/corr_spectra.npy', allow_pickle=True))
metab = pd.read_csv('HMDB/urine_metabolites.csv')
corr_metab = pd.read_csv('HMDB/corr_metabolites.csv')

all_spectra = corr_spectra + spectra
all_metab = corr_metab.append(metab, ignore_index = True)

new_metab = pd.DataFrame(columns = all_metab.columns)
true_spectra, decoy_spectra = dict(), dict()
for i in tqdm(range(len(all_spectra))):
    metabolite = all_metab.loc[i, 'Metabolite']
    precursor_mz = all_metab.loc[i, 'Precursor_mz']
    
    if metabolite in list(new_metab['Metabolite']):
        continue
    else:
        new_metab.loc[len(new_metab)] = all_metab.loc[i, :]
    
    if np.isnan(precursor_mz):
        true_spectra[metabolite] = all_spectra[i]['medium_energy']
        decoy_spectra[metabolite] = None
    else:
        decoy_spectrum = get_decoy_spectrum(precursor_mz, all_spectra[i]['medium_energy'])
        true_spectra[metabolite] = all_spectra[i]['medium_energy']
        decoy_spectra[metabolite] = decoy_spectrum
        
np.save('HMDB/true_urine_spectra.npy', true_spectra)
np.save('HMDB/decoy_urine_spectra.npy', decoy_spectra)
new_metab.to_csv('HMDB/all_metabolites.csv', index = False)
