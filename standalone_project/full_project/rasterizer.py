import numpy as np
from ctypes import *

from standalone_project.full_project.vector import vec2

rasterizer = cdll.LoadLibrary('./standalone_project/full_project/src_c/rasterizer.so')
rasterizer.test_read_write.argtypes = [c_int,
                                       np.ctypeslib.ndpointer(dtype=np.float32),
                                       np.ctypeslib.ndpointer(dtype=np.int32),
                                       np.ctypeslib.ndpointer(dtype=np.uint8)]

# rasterizer.test_read_write.restype = np.ctypeslib.ndpointer(dtype=np.uint8)
rasterizer.projector.argtypes = [c_int,
                                 np.ctypeslib.ndpointer(dtype=np.int32),
                                 np.ctypeslib.ndpointer(dtype=np.float32),
                                 np.ctypeslib.ndpointer(dtype=np.uint8), 
                                 c_float, 
                                 c_float, 
                                 c_float, 
                                 c_int]

def projector (len: c_int, labels: np.ctypeslib.ndpointer(dtype=np.int32), fp_vec:np.ctypeslib.ndpointer(dtype=np.float32), map_out:np.ctypeslib.ndpointer(dtype=np.uint8), mapcenter:vec2, mapsize: c_float, gridsize:c_int) -> None:
    rasterizer.projector(len, labels, fp_vec, map_out, c_float(mapcenter.x()), c_float(mapcenter.y()), mapsize, gridsize)

rasterizer.apply_BBA.argtypes = [c_int, #nFE
                                 c_int, #gridsize
                                 np.ctypeslib.ndpointer(dtype=np.float32), #FE
                                 np.ctypeslib.ndpointer(dtype=np.uint8), #mask
                                 np.ctypeslib.ndpointer(dtype=np.float32)] # map out

def apply_BBA (nFE:c_int, gridsize:c_int, FE:np.ctypeslib.ndpointer(dtype=np.float32), mask:np.ctypeslib.ndpointer(dtype=np.uint8), evid_map:np.ctypeslib.ndpointer(dtype=np.float32)) -> None:
    rasterizer.apply_BBA(nFE, gridsize, FE, mask, evid_map)

