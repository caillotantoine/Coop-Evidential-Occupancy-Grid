import numpy as np
from ctypes import *

metrics = cdll.LoadLibrary('/home/caillot/Documents/PhD/Projets/Coop-Evidential-Occupancy-Grid/standalone_project/full_project/src_c/metrics.so')


#  void TFPN(unsigned char *truth, unsigned char *test, int gridsize, unsigned char TFPN_sel, unsigned char label)
metrics.TFPN.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8), np.ctypeslib.ndpointer(dtype=np.uint8), c_int, c_char, c_char] 
def TFPN_w( truth:np.ctypeslib.ndpointer(dtype=np.uint8),  test:np.ctypeslib.ndpointer(dtype=np.uint8), gridsize:c_int, TFPN_sel:c_char, label:c_char):
    metrics.TFPN( truth,  test, gridsize, TFPN_sel, label) 


#  void toOccup(unsigned char *sem_map, unsigned char *out, int gridsize)
metrics.toOccup.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8), np.ctypeslib.ndpointer(dtype=np.uint8), c_int] 
def toOccup_w( sem_map:np.ctypeslib.ndpointer(dtype=np.uint8),  out:np.ctypeslib.ndpointer(dtype=np.uint8), gridsize:c_int):
    metrics.toOccup( sem_map,  out, gridsize) 