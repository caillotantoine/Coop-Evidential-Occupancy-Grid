import numpy as np
from ctypes import *

merger_cpp = cdll.LoadLibrary('./standalone_project/full_project/src_c/merger.so')

merger_cpp.mean_merger.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8),
                                   c_int, 
                                   c_int, 
                                   np.ctypeslib.ndpointer(dtype=np.float32)]
def mean_merger_fast(masks, gridsize) -> np.ndarray:
    out = np.zeros(shape=(gridsize, gridsize, 3), dtype=np.float32)
    masks_arr = np.array(masks)
    merger_cpp.mean_merger(masks_arr, gridsize, len(masks), out)
    return out

def merge(maps, gridsize, nFE) -> np.ndarray:
    evid_map_out = np.zeros(shape=(gridsize, gridsize, nFE), dtype=np.float32)

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



if __name__ == "__main__":
    v = [({"O"}, 0.1), ({"V"}, 0.6), ({"P"}, 0.0), ({"T"}, 0.0), ({"V", "P"}, 0.1), ({"V", "T"}, 0.1), ({"P", "T"}, 0.0), ({"V", "P", "T"}, 0.1)]
    p = [({'null'}, 0.1), ({'V'}, 0), ({'P'}, 0.6), ({'T'}, 0), ({'V', 'P'}, 0.1), ({'V', 'T'}, 0), ({'P', 'T'}, 0.1), ({'V', 'P', 'T'}, 0.1)]
    t = [({'null'}, 0.1), ({'V'}, 0), ({'P'}, 0), ({'T'}, 0.6), ({'V', 'P'}, 0), ({'V', 'T'}, 0.1), ({'P', 'T'}, 0.1), ({'V', 'P', 'T'}, 0.1)]
    u = [({'null'}, 0.1), ({'V'}, 0), ({'P'}, 0), ({'T'}, 0), ({'V', 'P'}, 0), ({'V', 'T'}, 0), ({'P', 'T'}, 0), ({'V', 'P', 'T'}, 0.9)]

    print(v[1][0].union(v[4][0]))