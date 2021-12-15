import ctypes
import multiprocessing
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
from multiprocessing import Pool
from merger import mean_merger, mean_merger_fast

import rasterizer

MAPSIZE = 120.0
GRIDSIZE = int(MAPSIZE) * 5

# fig, axes = plt.subplots(2, 2)


dataset_path:str = '/home/caillot/Documents/Dataset/CARLA_Dataset_B'
agents:List[Agent] = []
with open(f"{dataset_path}//information.json") as json_file:
    info = json.load(json_file)
    agent_l = info['agents']
    agents = [Agent(dataset_path=dataset_path, id=idx) for idx, agent in enumerate(agent_l) if agent['type'] != "pedestrian"]

for idx, agent in enumerate(agents):
    print(f"{idx} : \t{agent}")

def get_bbox(data):
    agent, frame = data
    return agent.get_visible_bbox(frame=frame)

def generate_evid_grid(agent_out):
    # agent, frame = data
    # agent_out = agent.get_visible_bbox(frame=frame)

    egg = EGG(mapsize=120.0, gridsize=(120*5))
    eggout = egg.projector_resterizer(agent_out)
    fp_poly = np.array([np.array([(v.get().T)[0] for v in poly], dtype=np.float32) for (poly, _) in eggout])
    fp_label = np.array([1 if label == 'vehicle' else 2 if label == 'pedestrian' else 3 if label == 'terrain' else 0 for (_, label) in eggout], dtype=
    np.int32)

    # Tested the speed between zeros() and empty(). Results were pretty similar. Thus we chose zeros for safety.
    mask = np.zeros(shape=(GRIDSIZE, GRIDSIZE), dtype=np.uint8)
    rasterizer.projector(len(fp_label), fp_label, fp_poly, mask, MAPSIZE, GRIDSIZE)

    nFE2 = 8
    FE2 = [[0.1, 0, 0, 0, 0, 0, 0, 0.9],
        [0.1, 0.6, 0, 0, 0.1, 0.1, 0, 0.1], 
        [0.1, 0, 0.6, 0, 0.1, 0, 0.1, 0.1], 
        [0.1, 0, 0, 0.6, 0, 0.1, 0.1, 0.1]]
    FE2 = np.array(FE2, dtype=np.float32)     
    evid_map2 = np.zeros(shape=(GRIDSIZE, GRIDSIZE, nFE2), dtype=np.float32)

    rasterizer.apply_BBA(nFE2, GRIDSIZE, FE2, mask, evid_map2)
    return (mask, evid_map2)


# a = agents[6]
pool = Pool(multiprocessing.cpu_count())
A_TODISP = -1 #     6 -> infrastructure
for frame in tqdm(range(10, 500)):
    data = [(agent, frame) for agent in agents]
    bboxes = pool.map(get_bbox, data)
    mask_eveid_maps = [generate_evid_grid(d) for d in bboxes]
    mask, evid_maps = zip(*mask_eveid_maps)
    mean_map = mean_merger_fast(mask, gridsize=GRIDSIZE)

    plt.imshow(mean_map)
    plt.pause(0.01)

    # emap = evid_maps[6]
    # # axes[0, 0].imshow(agents[6].get_rgb(frame=frame))
    # # axes[0, 0].set_title('Image')
    # axes[0, 0].imshow(mask[6])
    # axes[0, 0].set_title('Mask')
    # axes[0, 1].imshow(emap[:,:,1:4])
    # axes[0, 1].set_title('V, P, T')
    # axes[1, 0].imshow(emap[:,:,4:7])
    # axes[1, 0].set_title('VP, VT, PT')
    # axes[1, 1].imshow(emap[:,:,7])
    # axes[1, 1].set_title('VPT')
    # fig.suptitle(f'Frame {frame}')
    # plt.pause(0.01)

    
    








    # for a_idx, a in enumerate(agents):
    #     # if a.label == 'pedestrian':
    #     #     break
    #     agent_out = a.get_visible_bbox(frame=frame)

    #     egg = EGG(mapsize=120.0, gridsize=(120*5))
    #     # print('-------------------')
    #     eggout = egg.projector_resterizer(agent_out)
    #     fp_poly = np.array([np.array([(v.get().T)[0] for v in poly], dtype=np.float32) for (poly, _) in eggout])
    #     fp_label = np.array([1 if label == 'vehicle' else 2 if label == 'pedestrian' else 3 if label == 'terrain' else 0 for (_, label) in eggout], dtype=
    #     np.int32)
    #     # print(fp_poly)
    #     # print(fp_label)



    #     # Tested the speed between zeros() and empty(). Results were pretty similar. Thus we chose zeros for safety.
    #     mask = np.zeros(shape=(GRIDSIZE, GRIDSIZE), dtype=np.uint8)
    #     rasterizer.projector(len(fp_label), fp_label, fp_poly, mask, MAPSIZE, GRIDSIZE)
    #     if a_idx == A_TODISP:
    #         axes[0, 0].imshow(mask)
    #         axes[0, 0].set_title('Mask')
    #     # plt.imshow(mask)
    #     # plt.show()

    #     nFE = 4 # V, P, T, Ω
    #     FE = [[0.1, 0.1, 0.1, 0.7], # No observation
    #         [0.7, 0.1, 0.1, 0.1], # Vehicle
    #         [0.1, 0.7, 0.1, 0.1], # Pedestrian
    #         [0.1, 0.1, 0.7, 0.1]] # Terrain
    #     FE = np.array(FE, dtype=np.float32)
    #     evid_map = np.zeros(shape=(GRIDSIZE, GRIDSIZE, nFE), dtype=np.float32)

    #     nFE2 = 8
    #     FE2 = [[0.1, 0, 0, 0, 0, 0, 0, 0.9],
    #         [0.1, 0.6, 0, 0, 0.1, 0.1, 0, 0.1], 
    #         [0.1, 0, 0.6, 0, 0.1, 0, 0.1, 0.1], 
    #         [0.1, 0, 0, 0.6, 0, 0.1, 0.1, 0.1]]
    #     FE2 = np.array(FE2, dtype=np.float32)     
    #     evid_map2 = np.zeros(shape=(GRIDSIZE, GRIDSIZE, nFE2), dtype=np.float32)



    #     rasterizer.apply_BBA(nFE2, GRIDSIZE, FE2, mask, evid_map2)
    #     if a_idx == A_TODISP:
    #         axes[0, 1].imshow(evid_map2[:,:,1:4])
    #         axes[0, 1].set_title('V, P, T')
    #         axes[1, 0].imshow(evid_map2[:,:,4:7])
    #         axes[1, 0].set_title('VP, VT, PT')
    #         axes[1, 1].imshow(evid_map2[:,:,7])
    #         axes[1, 1].set_title('VPT')
    #         fig.suptitle(f'Frame {frame}')
    #         plt.pause(0.01)