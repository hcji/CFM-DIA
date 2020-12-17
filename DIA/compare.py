# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:05:08 2020

@author: hcji
"""

import numpy as np
from seq_encode.ms import ms_to_vec

converter = ms_to_vec()

def dot_similarty(q, r):
    q = np.array(q)
    r = np.array(r)
    qv = converter.transform(q)
    rv = converter.transform(r)
    sim = np.dot(qv, rv) / np.sqrt( np.dot(qv, qv) * np.dot(rv, rv) )
    return sim
    

def jaccard_similarity(q, r):
    q = np.array(q)
    r = np.array(r)
    ql = np.round(q[:,0], 2)
    rl = np.round(r[:,0], 2)
    intersection = len(list(set(ql).intersection(rl)))
    union = (len(ql) + len(ql)) - intersection
    sim = float(intersection / union)
    return sim