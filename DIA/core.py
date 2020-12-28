# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 08:26:33 2020

@author: hcji
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from tqdm import tqdm
from PyCFMID.PyCFMID import cfm_predict
from DIA.utils import load_data, get_precursor_eic, get_fragment_eic, get_peaks, get_decoy_spectrum

def one_sample_one_compound(data, precursor_mz, spectrum, mztol=0.1, peak_threshold=1000):
    precursor_eic = get_precursor_eic(data, precursor_mz, 0, mztol = 0.1, width = float('inf'))
    precursor_peaks, peaks_information = get_peaks(precursor_eic, peak_threshold=peak_threshold)
    
    fragment_mzs = spectrum['mz']
    precursor_rt, precursor_eic = precursor_eic[0], precursor_eic[1]
    fragment_rt, fragment_eics = get_fragment_eic(data, precursor_mz, fragment_mzs, 0, mztol = 0.1, width = float('inf'))
    
    output = dict()
    for i, rt in enumerate(precursor_peaks):
        width = max((peaks_information[1][i] - peaks_information[0][i]), 5)
        standard_rt = np.linspace(rt - 0.5 * width, rt + 0.5 * width, 100)
        precursor_eic1 = np.interp(standard_rt, precursor_rt, precursor_eic)
        '''
        fig = plt.figure(dpi = 300)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.plot(standard_rt, precursor_eic1, color = 'red', label = 'precursor')
        '''
        precursor_intensity = precursor_eic1[int(len(precursor_eic1) / 2)]
        fragment_intensity, fragment_correlation = [], []
        for j, fragment_eic in enumerate(fragment_eics):
            fragment_eic = np.interp(standard_rt, fragment_rt, fragment_eic)
            fragment_corr = pearsonr(precursor_eic1, fragment_eic)[0]
            # ax2.plot(standard_rt, fragment_eic, label = 'frag_{}'.format(round(fragment_mzs[j], 2)))
            fragment_intensity.append(fragment_eic[int(len(fragment_eic) / 2)])
            fragment_correlation.append(fragment_corr)
        '''
        ax1.set_ylabel('precursor intensity')
        ax2.set_ylim(0, 700000)
        ax2.set_ylabel('fragment intensity')
        ax1.set_xlabel('retention time')
        ax1.legend(loc = 'upper left')
        ax2.legend()
        plt.show()
        '''
        qv = np.array(fragment_intensity)
        wt = np.array(fragment_correlation)
        rv = np.array(spectrum)[:,1]
        dot_score = np.dot(qv, rv) / np.sqrt( np.dot(qv, qv) * np.dot(rv, rv) )
        cor_score = np.nansum(qv * wt) / np.nansum(qv)
        tot_score = 0.5 * (dot_score + cor_score)
        output[rt] = dict({'precursor_intensity':precursor_intensity, 'total_score': tot_score, 'ms2': np.array([spectrum.iloc[:,0], qv]).T})
    return output


def process_dataset(file_dir, file_met, file_spectra = None, energy = None, mztol=0.1, peak_threshold=1000, parallel=False, use_decoy=False):
    files = os.listdir(file_dir)
    metab = pd.read_csv(file_met)
    if energy == None:
        level = 'high_energy'
    elif energy <= 15:
        level = 'low_energy'
    elif energy <= 30:
        level = 'medium_energy'
    else:
        level = 'high_energy'
        
    if file_spectra == None:
        print('predicting spectra by CFM-ID \n')
        spectra = dict()
        for i, smi in enumerate(tqdm(metab['SMILES'])):
            precursor_mz = metab['Precursor_mz'][i]
            spectrum = cfm_predict(smi, prob_thresh=0.01, param_file='', config_file='', annotate_fragments=False, output_file=None, apply_postproc=True, suppress_exceptions=False)
            spectrum = spectrum[level]
            if use_decoy:
                spectrum = get_decoy_spectrum(precursor_mz, spectrum)
            spectra[metab['Metabolite'][i]] = spectrum
    else:
        spectra = np.load(file_spectra, allow_pickle=True).item()

    
    def process_one_file(f):
        f = file_dir + '/' + f
        data = load_data(f, energy)
        result_i = dict()
        for i in tqdm(range(len(metab))):
            metabolite = metab['Metabolite'][i]
            precursor_mz = metab['Precursor_mz'][i]
            spectrum = spectra[metabolite]
            if (spectrum is None) or (len(spectrum) < 1):
                continue
            res = one_sample_one_compound(data, precursor_mz, spectrum, mztol=mztol, peak_threshold=peak_threshold)
            if len(res) == 0:
                continue
            result_i[metabolite] = res
        return result_i
    
    
    print('processing dataset \n')
    results = dict()
    if not parallel:
        for f in files:
            f = file_dir + '/' + f
            print('processing ' + f)
            data = load_data(f, energy)
            result_i = dict()
            for i in tqdm(range(len(metab))):
                metabolite = metab['Metabolite'][i]
                precursor_mz = metab['Precursor_mz'][i]
                spectrum = spectra[metabolite]
                if (spectrum is None) or (len(spectrum) < 1):
                    continue
                res = one_sample_one_compound(data, precursor_mz, spectrum, mztol=mztol, peak_threshold=peak_threshold)
                if len(res) == 0:
                    continue
                result_i[metabolite] = res
            results[f] = result_i
    else:
        num_cores = multiprocessing.cpu_count()
        result_par = Parallel(n_jobs=num_cores, verbose=3)(delayed(process_one_file)(f) for f in files)
        for i, f in enumerate(files):
            results[f] = result_par[i]
    return results, spectra
        

def grouping_results(results, rt_tol = 5, n_candidate = 1000):
    samples = list(results.keys())
    columns = ['Sample', 'Metabolite','RT', 'Precursor_intensity', 'Fragment_mz', 'Fragment_intensity', 'Score']
    quant_table = []
    for s in tqdm(samples):
        r = results[s]
        met = list(r.keys())
        for m in met:
            ls = r[m]
            rts = np.array(list(ls.keys()))
            scores = np.array([ls[k]['total_score'] for k in ls.keys()])
            topn = np.argsort(-scores)[range(min(len(scores), n_candidate))]
            for rt in rts[topn]:
                sc = ls[rt]['total_score']
                pi = ls[rt]['precursor_intensity']
                for i in range(len(ls[rt]['ms2'])):
                    mz = ls[rt]['ms2'][i,0]
                    intensity = ls[rt]['ms2'][i,1]
                    quant_table.append([s, m, rt, pi, mz, intensity, sc])
    quant_table = pd.DataFrame(quant_table)
    quant_table.columns = columns
    
    group = 0
    n_group_table = []
    for m, sub in quant_table.groupby('Metabolite'):
        rts = sub['RT'].values
        inds = np.argsort(rts)
        rt0 = -float('inf')
        group_id = np.repeat(0, len(sub))
        for i in inds:
            rt = rts[i]
            if rt - rt0 > rt_tol:
                rt0 = rt
                group += 1
            group_id[i] = group
        sub['Group_id'] = group_id
        n_group_table.append(sub)
    n_group_table = pd.concat(n_group_table)
    return n_group_table


if __name__ == '__main__':
    pass
    