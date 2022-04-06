import numpy as np
from ctypes import *
from typing import List
from copy import deepcopy


decision = cdll.LoadLibrary('/home/caillot/Documents/PhD/Projets/Coop-Evidential-Occupancy-Grid/standalone_project/full_project/src_c/decision.so')


#  void decision(float *evid_map_in, unsigned char *sem_map, int gridsize, int nFE, int method)
decision.decision.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.uint8), c_int, c_int, c_int] 
def decision_w( evid_map_in:np.ctypeslib.ndpointer(dtype=np.float32),  sem_map:np.ctypeslib.ndpointer(dtype=np.uint8), gridsize:c_int, nFE:c_int, method:c_int):
    decision.decision( evid_map_in,  sem_map, gridsize, nFE, method) 

def cred2pign(evid_map_in:np.ctypeslib.ndpointer(dtype=np.float32), method:c_int = 0) -> np.ndarray:
    (gridsize, _, nFE) = evid_map_in.shape
    out_mask = np.zeros((gridsize, gridsize), dtype=np.uint8)
    decision_w(evid_map_in, out_mask, gridsize, nFE, method)
    return out_mask
