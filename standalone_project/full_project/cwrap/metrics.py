import numpy as np
from ctypes import *

metrics = cdll.LoadLibrary('/home/caillot/Documents/PhD/Projets/Coop-Evidential-Occupancy-Grid/standalone_project/full_project/src_c/metrics.so')


# #  void TFPN(unsigned char *truth, unsigned char *test, int gridsize, unsigned char TFPN_sel, unsigned char label)
# metrics.TFPN.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8), np.ctypeslib.ndpointer(dtype=np.uint8), c_int, c_char, c_char] 
# def TFPN( truth:np.ctypeslib.ndpointer(dtype=np.uint8),  test:np.ctypeslib.ndpointer(dtype=np.uint8), gridsize:c_int, TFPN_sel:c_char, label:c_char):
#     metrics.TFPN( truth,  test, gridsize, TFPN_sel, label) 

    #  int TFPN(unsigned char *truth, unsigned char *test, int gridsize, unsigned char TFPN_sel, unsigned char label)
metrics.TFPN.restype = c_int
metrics.TFPN.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8), np.ctypeslib.ndpointer(dtype=np.uint8), np.ctypeslib.ndpointer(dtype=np.uint8), c_int, c_int, c_char, c_char] 
def TFPN( truth:np.ctypeslib.ndpointer(dtype=np.uint8),  test:np.ctypeslib.ndpointer(dtype=np.uint8), zones: np.ctypeslib.ndpointer(dtype=np.uint8), coop_lvl:c_char, gridsize:c_int, TFPN_sel:c_char, label:c_char):
        return metrics.TFPN( truth, test, zones, coop_lvl, gridsize, TFPN_sel, label) 


#  void toOccup(unsigned char *sem_map, unsigned char *out, int gridsize)
metrics.toOccup.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8), np.ctypeslib.ndpointer(dtype=np.uint8), c_int] 
def toOccup_w( sem_map:np.ctypeslib.ndpointer(dtype=np.uint8),  out:np.ctypeslib.ndpointer(dtype=np.uint8), gridsize:c_int):
    metrics.toOccup( sem_map,  out, gridsize) 

def toOccup(sem_map:np.ctypeslib.ndpointer(dtype=np.uint8), gridsize:c_int) -> np.ndarray:
    out = np.zeros(shape=(gridsize, gridsize), dtype=np.uint8)
    toOccup_w(sem_map=sem_map, out=out, gridsize=gridsize)
    return out