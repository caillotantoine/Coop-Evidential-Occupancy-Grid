import numpy as np
from ctypes import *
from typing import List
from copy import deepcopy

merger = cdll.LoadLibrary('./standalone_project/full_project/src_c/merger.so')

#  void mean_merger(unsigned char *masks, int gridsize, int n_agents, float *out)
merger.mean_merger.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8), c_int, c_int, np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32), c_int] 
def mean_merger_w( masks:np.ctypeslib.ndpointer(dtype=np.uint8), gridsize:c_int, n_agents:c_int,  out:np.ctypeslib.ndpointer(dtype=np.float32), FE:np.ctypeslib.ndpointer(dtype=np.float32), nFE:c_int):
	merger.mean_merger( masks, gridsize, n_agents,  out, np.array(FE).astype(np.float32), nFE) 



def mean_merger_fast(masks, gridsize, FE) -> np.ndarray:
    out = np.zeros(shape=(gridsize, gridsize, 3), dtype=np.float32)
    masks_arr = np.stack(masks, axis=2)
    mean_merger_w(masks_arr, gridsize, len(masks), out, FE, len(FE[1]))
    return out

def mean_merger(masks, gridsize) -> np.ndarray:
    out = np.zeros(shape=(gridsize, gridsize, 3), dtype=np.float32)
    for mask in masks:
        for i in range(0, gridsize):
            for j in range(0, gridsize):
                if mask[i][j] == int('0b01000000', 2):      # Vehicle
                    out[i][j] += [1.0, 0.0, 0.0]
                elif mask[i][j] == int('0b10000000', 2): # Pedestrian
                    out[i][j] += [0.0, 1.0, 0.0]
                elif mask[i][j] == int('0b00000010', 2): # Terrain
                    out[i][j] += [0.0, 0.0, 1.0]
                else:
                    out[i][j] += [0.5, 0.5, 0.5]
    return out / float(len(masks))

#  void DST_merger(float *evid_maps_in, float *inout, int gridsize, int nFE, int n_agents, unsigned char method)
merger.DST_merger.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32), c_int, c_int, c_int, c_char] 
def DST_merger_w( evid_maps_in:np.ctypeslib.ndpointer(dtype=np.float32),  inout:np.ctypeslib.ndpointer(dtype=np.float32), gridsize:c_int, nFE:c_int, n_agents:c_int, method:c_char):
	merger.DST_merger( evid_maps_in,  inout, gridsize, nFE, n_agents, method)

merger.DST_merger_CUDA.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32), c_int, c_int, c_int, c_char] 
def DST_merger_CUDA_w( evid_maps_in:np.ctypeslib.ndpointer(dtype=np.float32),  inout:np.ctypeslib.ndpointer(dtype=np.float32), gridsize:c_int, nFE:c_int, n_agents:c_int, method:c_char):
	merger.DST_merger_CUDA( evid_maps_in,  inout, gridsize, nFE, n_agents, method)

def DST_merger(evid_maps:List[np.ndarray], gridsize, CUDA:bool = False, method:int = 0) -> np.ndarray:
    # evid_maps_l = evid_maps
    inout = deepcopy(evid_maps.pop(0))
    if len(evid_maps) == 0:
        return inout
    evid_maps_arr = np.stack(evid_maps, axis = 2) # [gridsize][gridsize][n_agents][n_elements]
    # evid_maps_arr = evid_maps_arr
    nFE = evid_maps_arr.shape[3]
    # inout = np.zeros(shape=(gridsize, gridsize, nFE), dtype=np.float32)
    if CUDA:
        DST_merger_CUDA_w(evid_maps_in=evid_maps_arr, inout=inout, gridsize=gridsize, nFE=nFE, n_agents=len(evid_maps), method=c_char(method))
    else:
        DST_merger_w(evid_maps_in=evid_maps_arr, inout=inout, gridsize=gridsize, nFE=nFE, n_agents=len(evid_maps), method=c_char(method))
    return inout
    



# if __name__ == "__main__":
#     v = [({"O"}, 0.1), ({"V"}, 0.6), ({"P"}, 0.0), ({"T"}, 0.0), ({"V", "P"}, 0.1), ({"V", "T"}, 0.1), ({"P", "T"}, 0.0), ({"V", "P", "T"}, 0.1)]
#     p = [({'null'}, 0.1), ({'V'}, 0), ({'P'}, 0.6), ({'T'}, 0), ({'V', 'P'}, 0.1), ({'V', 'T'}, 0), ({'P', 'T'}, 0.1), ({'V', 'P', 'T'}, 0.1)]
#     t = [({'null'}, 0.1), ({'V'}, 0), ({'P'}, 0), ({'T'}, 0.6), ({'V', 'P'}, 0), ({'V', 'T'}, 0.1), ({'P', 'T'}, 0.1), ({'V', 'P', 'T'}, 0.1)]
#     u = [({'null'}, 0.1), ({'V'}, 0), ({'P'}, 0), ({'T'}, 0), ({'V', 'P'}, 0), ({'V', 'T'}, 0), ({'P', 'T'}, 0), ({'V', 'P', 'T'}, 0.9)]

#     print(v[1][0].union(v[4][0]))




