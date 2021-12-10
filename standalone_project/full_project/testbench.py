import ctypes
import numpy as np
from agent import Agent
from vector import vec2, vec3, vec4
from Tmat import TMat
from bbox import Bbox2D, Bbox3D
from tqdm import tqdm
from plucker import plkrPlane
from projector import project_BBox2DOnPlane
import json
from typing import List
from EGG import EGG
from ctypes import *
import matplotlib.pyplot as plt

MAPSIZE = 120.0
GRIDSIZE = int(MAPSIZE) * 5


dataset_path:str = '/home/caillot/Documents/Dataset/CARLA_Dataset_B'
agents:List[Agent] = []
with open(f"{dataset_path}//information.json") as json_file:
    info = json.load(json_file)
    agent_l = info['agents']
    agents = [Agent(dataset_path=dataset_path, id=idx) for idx, agent in enumerate(agent_l) if agent['type'] != "pedestrian"]

for idx, agent in enumerate(agents):
    print(f"{idx} : \t{agent}")


a = agents[6]
agent_out = a.get_visible_bbox(frame=56)



egg = EGG(mapsize=120.0, gridsize=(120*5))
print('-------------------')
eggout = egg.projector_resterizer(agent_out)
fp_poly = np.array([np.array([(v.get().T)[0] for v in poly], dtype=np.float32) for (poly, _) in eggout])
fp_label = np.array([1 if label == 'vehicle' else 2 if label == 'pedestrian' else 0 for (_, label) in eggout], dtype=
np.int32)
print(fp_poly)
print(fp_label)

rasterizer = cdll.LoadLibrary('./standalone_project/full_project/src_c/rasterizer.so')
rasterizer.test_read_write.argtypes = [ctypes.c_int,
                                       np.ctypeslib.ndpointer(dtype=np.float32),
                                       np.ctypeslib.ndpointer(dtype=np.int32),
                                       np.ctypeslib.ndpointer(dtype=np.uint8)]
# rasterizer.test_read_write.restype = np.ctypeslib.ndpointer(dtype=np.uint8)
rasterizer.projector.argtypes = [ctypes.c_int,
                                 np.ctypeslib.ndpointer(dtype=np.int32),
                                 np.ctypeslib.ndpointer(dtype=np.float32),
                                 np.ctypeslib.ndpointer(dtype=np.uint8), 
                                 ctypes.c_float, 
                                 ctypes.c_int]



# Tested the speed between zeros() and empty(). Results were pretty similar. Thus we chose zeros for safety.
map = np.zeros(shape=(GRIDSIZE, GRIDSIZE), dtype=np.uint8)
rasterizer.projector(len(fp_label), fp_label, fp_poly, map, MAPSIZE, GRIDSIZE)
plt.imshow(map)
plt.show()

