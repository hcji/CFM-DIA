# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:16:21 2021

@author: hcji
"""


import numpy as np
import matplotlib.pyplot as plt

from PyCFMID.PyCFMID import cfm_predict
from DIA.core import one_sample_one_compound
from DIA.utils import load_data, get_fragment_eic, get_precursor_eic, get_ms2

file = 'D:/data/MTBLS816_mzML/170516_S1_ZHP_ETYPE_B1_021_QC1.mzXML'

# compound 1: Asparagine
# compound 2: Serine
# compound 3: N-Acetylputrescine
smi1 = '[H]OC(=O)C([H])(N([H])[H])C([H])([H])C(=O)N([H])[H]'
smi2 = 'C([C@@H](C(=O)O)N)O'
smi3 = '[H]N([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])N([H])C(=O)C([H])([H])[H]'

rt = 547
precursor_mz1 = 133.0611
precursor_mz2 = 106.0499
precursor_mz3 = 131.1183

data = load_data(file, 10)
spectrum1 = cfm_predict(smi1, prob_thresh=0.001, param_file='', config_file='', annotate_fragments=False, output_file=None, apply_postproc=True, suppress_exceptions=False)
spectrum1 = spectrum1['low_energy']
spectrum1 = spectrum1[spectrum1['intensity'] > 5]
spectrum1 = spectrum1[spectrum1['mz'] < precursor_mz1-1]

spectrum2 = cfm_predict(smi2, prob_thresh=0.001, param_file='', config_file='', annotate_fragments=False, output_file=None, apply_postproc=True, suppress_exceptions=False)
spectrum2 = spectrum2['low_energy']
spectrum2 = spectrum2[spectrum2['intensity'] > 5]
spectrum2 = spectrum2[spectrum2['mz'] < precursor_mz2-1]

spectrum3 = cfm_predict(smi3, prob_thresh=0.001, param_file='', config_file='', annotate_fragments=False, output_file=None, apply_postproc=True, suppress_exceptions=False)
spectrum3 = spectrum3['low_energy']
spectrum3 = spectrum3[spectrum3['intensity'] > 5]
spectrum3 = spectrum3[spectrum3['mz'] < precursor_mz3-1]

frag_eic1 = get_fragment_eic(data, precursor_mz1, spectrum1['mz'], rt, mztol = 0.01, width = 10)
frag_eic2 = get_fragment_eic(data, precursor_mz2, spectrum2['mz'], rt, mztol = 0.01, width = 10)
frag_eic3 = get_fragment_eic(data, precursor_mz3, spectrum3['mz'], rt, mztol = 0.01, width = 10)

markers = ['v', '+', '.', '*', '^']
plt.figure(dpi = 300, figsize = (6,12))
for i in range(len(frag_eic1[1])):
    plt.plot(frag_eic1[0], frag_eic1[1][i], label = 'Asn: ' + str(round(spectrum1['mz'].iloc[i], 2)), color = 'red', marker = markers[i])
for i in range(len(frag_eic2[1])):
    plt.plot(frag_eic2[0], frag_eic2[1][i], label = 'Ser: ' + str(round(spectrum2['mz'].iloc[i], 2)), color = 'darkgreen', marker = markers[i])
for i in range(len(frag_eic3[1])):
    plt.plot(frag_eic3[0], frag_eic3[1][i], label = 'N-Ace:' + str(round(spectrum3['mz'].iloc[i], 2)), color = 'navy', marker = markers[i])
plt.xlabel('retention time')
plt.ylabel('intensity')
plt.legend()
plt.show()


prec_eic1 = get_precursor_eic(data, precursor_mz1, rt, mztol = 0.01, width = 20)
prec_eic2 = get_precursor_eic(data, precursor_mz2, rt, mztol = 0.01, width = 20)
prec_eic3 = get_precursor_eic(data, precursor_mz3, rt, mztol = 0.01, width = 20)

plt.figure(dpi = 300, figsize = (8, 6))
plt.plot(prec_eic1[0], prec_eic1[1], label = 'Asn', color = 'red')
plt.plot(prec_eic2[0], prec_eic2[1], label = 'Ser', color = 'darkgreen')
plt.plot(prec_eic3[0], prec_eic3[1], label = 'N-Ace', color = 'navy')
plt.xlabel('retention time')
plt.ylabel('intensity')
plt.legend()
plt.show()


ms2 = get_ms2(data, precursor_mz1, rt)
plt.figure(dpi = 300, figsize = (15, 5))
plt.vlines(ms2[:,0], np.zeros(len(ms2)), ms2[:,1], color='black')
plt.axhline(0, color='black')
plt.xlim(0, 400)
plt.xlabel('m/z')
plt.ylabel('intensity')
plt.show()


spectrum = cfm_predict(smi3, prob_thresh=0.001, param_file='', config_file='', annotate_fragments=False, output_file=None, apply_postproc=True, suppress_exceptions=False)
spectrum = spectrum['low_energy']
res = one_sample_one_compound(data, precursor_mz1, spectrum, mztol=0.01, peak_threshold=5000)