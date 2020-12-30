#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:34:30 2020

@author: hcji
"""

import pyopenms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bisect import bisect_left, bisect_right
from scipy.signal import find_peaks, savgol_filter
from DIA.peak_eval import PeakEval

def load_data(mzml, select_energy=None):
    exp = pyopenms.MSExperiment()
    if mzml.split('.')[-1] == 'mzML':
        pyopenms.MzMLFile().load(mzml, exp)
    elif mzml.split('.')[-1] == 'mzXML':
        pyopenms.MzXMLFile().load(mzml, exp)
    else:
        raise IOError('unsupported format')
     
    size = exp.getNrSpectra()   
    peak_data = dict()
    for i in range(size):
        p = exp.getSpectrum(i)
        if p.getMSLevel() == 1:
            lower = -1
            upper = -1
        else:
            if select_energy != None:
                if p.getPrecursors()[0].getMetaValue('collision energy') != select_energy:
                    continue
            lower = round(p.getPrecursors()[0].getMZ() - p.getPrecursors()[0].getIsolationWindowLowerOffset(), 0)
            upper = round(p.getPrecursors()[0].getMZ() + p.getPrecursors()[0].getIsolationWindowUpperOffset(), 0)
        n = str(lower) + '_' + str(upper)
        if n not in peak_data.keys():
            peak_data[n] = []
        rt = p.getRT()
        peaks = p.get_peaks()
        peak_data[n].append({'rt': rt, 'mz': peaks[0], 'intensity':peaks[1]})
    
    return peak_data


def get_precursor_eic(data, pmz, rt, mztol = 0.1, width = 30): 
    swath = np.array([d.split('_') for d in data.keys()]).astype(float)
    
    wh1 = np.where(swath[:,0] == -1)[0]
    wh2 = np.where(swath[:,1] == -1)[0]
    wh = np.intersect1d(wh1, wh2)

    if len(wh) == 1:
        peaks = data[list(data.keys())[wh[0]]]
    else:
        peaks = []
        for w in wh:
            peaks += data[list(data.keys())[w]]
    
    rts = [float(p['rt']) for p in peaks]
    rtrange = [rt - width, rt + width]
    wh = np.arange(bisect_left(rts, rtrange[0]), bisect_right(rts, rtrange[1]))
    
    rts = np.array(rts)[wh]
    peaks = np.array(peaks)[wh]
    
    ind = np.argsort(rts)
    rts = rts[ind]
    peaks = peaks[ind]
    
    eic = []
    mzrange = [pmz - mztol, pmz + mztol]
    for p in peaks:
        mzs = p['mz']
        intensities = p['intensity']
        sel_peaks = np.arange(bisect_left(mzs, mzrange[0]), bisect_right(mzs, mzrange[1]))
        eic.append(np.sum(intensities[sel_peaks]))
    eic = np.array(eic)
    eic = savgol_filter(eic, min(9, len(eic)), 2)
    eic[eic < 0] = 0 
    return rts, eic


def get_fragment_eic(data, pmz, fmzs, rt, mztol = 0.1, width = 30): 
    swath = np.array([d.split('_') for d in data.keys()]).astype(float)
    
    wh1 = np.where(swath[:,0] < pmz)[0]
    wh2 = np.where(swath[:,1] > pmz)[0]
    wh = np.intersect1d(wh1, wh2)
    if len(wh) == 0:
        wh = np.argmin(np.abs((swath[:,0] + swath[:,1]) / 2 - pmz))
        peaks = data[list(data.keys())[wh]]
    elif len(wh) == 1:
        peaks = data[list(data.keys())[wh[0]]]
    else:
        peaks = []
        for w in wh:
            peaks += data[list(data.keys())[w]]
    rts = np.array([float(p['rt']) for p in peaks])
    peaks = np.array(peaks)
    
    ind = np.argsort(rts)
    rts = rts[ind]
    peaks = peaks[ind]
    
    rtrange = [rt - width, rt + width]
    wh = np.arange(bisect_left(rts, rtrange[0]), bisect_right(rts, rtrange[1]))
    rts = rts[wh]
    peaks = peaks[wh]
    
    eics = []
    for fmz in fmzs:
        mzrange = [fmz - mztol, fmz + mztol]
        eic = []
        for p in peaks:
            mzs = p['mz']
            intensities = p['intensity']
            sel_peaks = np.arange(bisect_left(mzs, mzrange[0]), bisect_right(mzs, mzrange[1]))
            eic.append(np.sum(intensities[sel_peaks]))
        eic = np.array(eic)
        eic = savgol_filter(eic, min(9, len(eic)), 2)
        eic[eic < 0] = 0
        # plt.plot(rts, eic)
        eics.append(eic)
    return rts, eics


def get_ms2(data, pmz, exrt):
    swath = np.array([d.split('_') for d in data.keys()]).astype(float)
    wh1 = np.where(swath[:,0] < pmz)[0]
    wh2 = np.where(swath[:,1] > pmz)[0]
    wh = np.intersect1d(wh1, wh2)
    if len(wh) > 0:
        wh = wh[0]
    else:
        wh = np.argmin(np.abs(swath[:,0] - pmz))
        
    peaks = data[list(data.keys())[wh]]
    rts = [float(p['rt']) for p in peaks]
    wh = np.argmin(np.abs(rts - exrt))
    return np.array([peaks[wh]['mz'], peaks[wh]['intensity']]).T


def get_peaks(eic, width=5, distance=5, peak_threshold=None, gaussian_score=0.7):
    x = eic[0]
    y = eic[1]
    z = np.array([x, y]).T
    if peak_threshold == None:
        threshold = np.percentile(y, 0.95)
    else:
        threshold = max(np.percentile(y, 0.95), peak_threshold)
    pks = find_peaks(y, width=width, distance=distance)
    keep = np.where(y[pks[0]] > threshold)[0]
    center, left, right, score = [], [], [], []
    for k in keep:
        ind = pks[0][k]
        w = pks[1]['widths'][k]
        l = max(0, ind - 2 * int(w))
        r = min(len(x)-1, ind + 2 * int(w))
        gs = PeakEval(z, x[l], x[r]).GaussianSimilarity()
        if gs < gaussian_score:
            continue
        center.append(x[ind])
        left.append(x[l])
        right.append(x[r])
        score.append(gs)
    return center, (left, right, score)


def get_decoy_spectrum(pmz, spectrum):
    mz = pmz - spectrum['mz']
    intensity = spectrum['intensity']
    keep = np.where(mz > 1)[0]
    output = pd.DataFrame({'mz': mz[keep], 'intensity': intensity[keep]})
    output = output.sort_values('mz', )
    return output

'''
def normalize_eic(eics):
    return np.array([l / sum(l) for l in eics])
'''

if __name__ == '__main__':
    pass
    