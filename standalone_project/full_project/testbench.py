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
from merger import mean_merger, mean_merger_fast, DST_merger
from decision import cred2pign

import rasterizer

MAPSIZE = 120.0
GRIDSIZE = int(MAPSIZE) * 5



# FE = [[0.1, 0, 0, 0, 0, 0, 0, 0.9],
#     [0.1, 0.6, 0, 0, 0.1, 0.1, 0, 0.1], 
#     [0.1, 0, 0.6, 0, 0.1, 0, 0.1, 0.1], 
#     [0.1, 0, 0, 0.6, 0, 0.1, 0.1, 0.1]]

fig, axes = plt.subplots(2, 3)
SAVE_PATH = '/home/caillot/Documents/output_algo/'
CPT_MEAN = False
ALGO = 'Conjonctive'
ALGOID = 1


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

    nFE = 8
    #      Ø    V    P    VP   T    VT   PT   VPT
    FE = [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9], # VPT
        [0.1, 0.6, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1], # V
        [0.1, 0.0, 0.6, 0.1, 0.0, 0.0, 0.1, 0.1], # P
        [0.1, 0.0, 0.0, 0.0, 0.6, 0.1, 0.1, 0.1]] # T

    FE = np.array(FE, dtype=np.float32)     
    evid_map = np.zeros(shape=(GRIDSIZE, GRIDSIZE, nFE), dtype=np.float32)

    rasterizer.apply_BBA(nFE, GRIDSIZE, FE, mask, evid_map)
    return (mask, evid_map)





dataset_path:str = '/home/caillot/Documents/Dataset/CARLA_Dataset_B'
agents:List[Agent] = []
with open(f"{dataset_path}//information.json") as json_file:
    info = json.load(json_file)
    agent_l = info['agents']
    agents = [Agent(dataset_path=dataset_path, id=idx) for idx, agent in enumerate(agent_l) if agent['type'] != "pedestrian"]

for idx, agent in enumerate(agents):
    print(f"{idx} : \t{agent}")


# a = agents[6]
pool = Pool(multiprocessing.cpu_count())
A_TODISP = -1 #     6 -> infrastructure
for frame in tqdm(range(10, 500)):
    data = [(agent, frame) for agent in agents]
    bboxes = pool.map(get_bbox, data)
    mask_eveid_maps = [generate_evid_grid(d) for d in bboxes]
    mask, evid_maps = zip(*mask_eveid_maps)

    if CPT_MEAN:
        mean_map = mean_merger_fast(mask, gridsize=GRIDSIZE)
        sem_map_mean = cred2pign(mean_map, method=-1)
        plt.imsave(f'{SAVE_PATH}/Mean/RAW/{frame:06d}.png', mean_map)
        plt.imsave(f'{SAVE_PATH}/Mean/SEM/{frame:06d}.png', sem_map_mean)


    evid_out = DST_merger(evid_maps=list(evid_maps), gridsize=GRIDSIZE, CUDA=False, method=ALGOID)
    plt.imsave(f'{SAVE_PATH}/{ALGO}/RAW/{frame:06d}-v-p-t.png', evid_out[:,:,[1, 2, 4]])
    plt.imsave(f'{SAVE_PATH}/{ALGO}/RAW/{frame:06d}-vp-vt-pt.png', evid_out[:,:,[3, 5, 6]])
    plt.imsave(f'{SAVE_PATH}/{ALGO}/RAW/{frame:06d}-vpt.png', evid_out[:,:,7])
    for m in range(3):
        sem_map = cred2pign(evid_out, method=m)
        plt.imsave(f'{SAVE_PATH}/{ALGO}/{m}/{frame:06d}.png', sem_map)


    # plt.imshow(mean_map)
    # plt.pause(0.01)

    # emap = evid_maps[6]
    # axes[0, 0].imshow(agents[6].get_rgb(frame=frame))
    # axes[0, 0].set_title('Image')

    # axes[0, 0].imshow(mean_map)
    # axes[0, 0].set_title('Mask')
    # axes[0, 1].imshow(evid_out[:,:,[1, 2, 4]])
    # axes[0, 1].set_title('V, P, T')
    # axes[0, 2].imshow(evid_out[:,:,[3, 5, 6]])
    # axes[0, 2].set_title('VP, VT, PT')
    # axes[1, 1].imshow(evid_out[:,:,7])
    # axes[1, 1].set_title('VPT')
    # # axes[1, 0].imshow(evid_out[:,:,4:7])
    # # axes[1, 0].set_title('VP, VT, PT')
    # axes[1, 2].imshow(sem_map)
    # axes[1, 2].set_title('sem_map evid')
    # # axes[1, 0].imshow(sem_map_mean)
    # # axes[1, 0].set_title('sem_map mean')
    # fig.suptitle(f'Frame {frame}')
    # plt.pause(0.01)