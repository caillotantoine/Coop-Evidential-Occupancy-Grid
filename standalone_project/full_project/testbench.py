import ctypes
import multiprocessing
from turtle import color
import numpy as np
from agent import Agent
from vector import vec2, vec3, vec4
from Tmat import TMat
from bbox import Bbox2D, Bbox3D
from tqdm import tqdm
from plucker import plkrPlane
from projector import project_BBox2DOnPlane
import json
from typing import List, Tuple
from EGG import EGG
from ctypes import *
import matplotlib.pyplot as plt
from multiprocessing import Pool
from merger import mean_merger, mean_merger_fast, DST_merger
from decision import cred2pign
import cv2 as cv

import rasterizer

MAPSIZE = 120.0
STEPGRID = 5
GRIDSIZE = int(MAPSIZE) * STEPGRID




# FE = [[0.1, 0, 0, 0, 0, 0, 0, 0.9],
#     [0.1, 0.6, 0, 0, 0.1, 0.1, 0, 0.1], 
#     [0.1, 0, 0.6, 0, 0.1, 0, 0.1, 0.1], 
#     [0.1, 0, 0, 0.6, 0, 0.1, 0.1, 0.1]]

fig, axes = plt.subplots(2, 3)
SAVE_PATH = '/home/caillot/Documents/output_algo/'
CPT_MEAN = True
ALGO = 'Dempster'
ALGOID = 0

ANTOINE_M = False


def get_bbox(data):
    agent, frame = data
    return agent.get_visible_bbox(frame=frame)

def get_pred(data):
    agent, frame = data
    return agent.get_pred(frame=frame)

def generate_evid_grid(agent_out:Tuple[List[Bbox2D], TMat, TMat] = None, agent_3D:List[Bbox3D] = None, antoine=False):
    # agent, frame = data
    # agent_out = agent.get_visible_bbox(frame=frame)

    egg = EGG(mapsize=MAPSIZE, gridsize=(GRIDSIZE))
    mask = np.zeros(shape=(GRIDSIZE, GRIDSIZE), dtype=np.uint8)

    if agent_out != None:
        eggout = egg.projector_resterizer(agent_out)
        fp_poly = np.array([np.array([(v.get().T)[0] for v in poly], dtype=np.float32) for (poly, _) in eggout])
        fp_label = np.array([1 if label == 'vehicle' else 2 if label == 'pedestrian' else 3 if label == 'terrain' else 0 for (_, label) in eggout], dtype=
        np.int32)
        rasterizer.projector(len(fp_label), fp_label, fp_poly, mask, MAPSIZE, GRIDSIZE)
    elif agent_3D != None:
        if antoine == False:
            mask[:,:] = int('0b00000010', 2)
        for agent in agent_3D:
            bboxsize = agent.get_size()
            label = agent.get_label()
            poseT = agent.get_TPose()
            # print(f'{agent.myid} of a size of {agent.get_bbox3d()} at {agent.get_pose()}')
            # bin_mask = [np.array([1.0, 1.0, 0.0, 1.0]), np.array([1.0, -1.0, 0.0, 1.0]), np.array([-1.0, -1.0, 0.0, 1.0]), np.array([-1.0, 1.0, 0.0, 1.0])]
            if antoine:
                bin_mask = [np.array([0.5, -0.5, 0.5, 1.0]), np.array([0.5, -0.5, -0.5, 1.0]), np.array([-0.5, -0.5, -0.5, 1.0]), np.array([-0.5, -0.5, 0.5, 1.0])]
            else:
                bin_mask = [np.array([0.5, 0.5, 0.0, 1.0]), np.array([0.5, -0.5, 0.0, 1.0]), np.array([-0.5, -0.5, 0.0, 1.0]), np.array([-0.5, 0.5, 0.0, 1.0])]
            
            # print(f'{label} of size {bboxsize} at:\n{poseT}')
            centered_fps = [(bboxsize.vec4().get().T * m).T for m in bin_mask]
            # print(centered_fps[0].shape)

            # print(poseT)
            fps = [poseT @ v for v in centered_fps]
            # print(fps)
            if antoine:
                fps_pix = np.array([np.array((((v * STEPGRID) + (GRIDSIZE / 2)).T)[0][[0, 2]], dtype=int) for v in fps])
            else:
                fps_pix = np.array([np.array((((v * STEPGRID) + (GRIDSIZE / 2)).T)[0][:2], dtype=int) for v in fps])
            # print(fps_pix)
            cv.fillPoly(mask, pts=[fps_pix], color=(int('0b01000000', 2) if label == 'vehicle' else int('0b10000000', 2)))
    else: 
        raise NameError("Both agent_out and agent_3D are set to None. Assign a value to at least one of them.")
        

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
    agents2gndtruth = [Agent(dataset_path=dataset_path, id=idx) for idx, agent in enumerate(agent_l) if agent['type'] != "infrastructure"]

for idx, agent in enumerate(agents):
    print(f"{idx} : \t{agent}")
    


# a = agents[6]
pool = Pool(multiprocessing.cpu_count())
A_TODISP = -1 #     6 -> infrastructure
for frame in tqdm(range(10, 500)):
    data = [(agent, frame) for agent in agents]
    if ANTOINE_M:
        # bboxes = pool.map(get_pred, data)
        bboxes = [a.get_pred(frame) for a in agents]
        # print(bboxes)
        mask_eveid_maps = [generate_evid_grid(agent_3D=d, antoine=True) for d in bboxes]
    else:
        bboxes = pool.map(get_bbox, data)
        mask_eveid_maps = [generate_evid_grid(agent_out=d) for d in bboxes]
    mask, evid_maps = zip(*mask_eveid_maps)
    # print(mask)

    gnd_agent = [agent.get_state(frame).get_bbox3d() for agent in agents2gndtruth]
    mask_eveid_maps_GND = generate_evid_grid(agent_3D=gnd_agent)
    (mask_GND, evid_maps_GND) = mask_eveid_maps_GND

    if CPT_MEAN:
        mean_map = mean_merger_fast(mask, gridsize=GRIDSIZE)
        sem_map_mean = cred2pign(mean_map, method=-1)
        # plt.imsave(f'{SAVE_PATH}/Mean/RAW/{frame:06d}.png', mean_map)
        # plt.imsave(f'{SAVE_PATH}/Mean/SEM/{frame:06d}.png', sem_map_mean)


    evid_out = DST_merger(evid_maps=list(evid_maps), gridsize=GRIDSIZE, CUDA=False, method=ALGOID)
    # plt.imsave(f'{SAVE_PATH}/{ALGO}/RAW/{frame:06d}-v-p-t.png', evid_out[:,:,[1, 2, 4]])
    # plt.imsave(f'{SAVE_PATH}/{ALGO}/RAW/{frame:06d}-vp-vt-pt.png', evid_out[:,:,[3, 5, 6]])
    # plt.imsave(f'{SAVE_PATH}/{ALGO}/RAW/{frame:06d}-vpt.png', evid_out[:,:,7])
    for m in range(3):
        sem_map = cred2pign(evid_out, method=3)
        plt.imsave(f'{SAVE_PATH}/{ALGO}/{m}/{frame:06d}.png', sem_map)


    # plt.imshow(mean_map)
    # plt.pause(0.01)

    # emap = evid_maps[6]
    # axes[0, 0].imshow(agents[6].get_rgb(frame=frame))
    # axes[0, 0].set_title('Image')

    axes[0, 0].imshow(np.array(mask)[[6, 7, 8]].transpose(1, 2, 0))
    axes[0, 0].set_title('Mask')
    axes[0, 1].imshow(evid_out[:,:,[1, 2, 4]])
    axes[0, 1].set_title('V, P, T')
    axes[0, 2].imshow(evid_out[:,:,[3, 5, 6]])
    axes[0, 2].set_title('VP, VT, PT')
    axes[1, 1].imshow(sem_map_mean)
    axes[1, 1].set_title('Sem map mean')
    axes[1, 0].imshow(mask_GND)
    axes[1, 0].set_title('Ground truth')
    # axes[1, 0].imshow(evid_out[:,:,4:7])
    # axes[1, 0].set_title('VP, VT, PT')
    axes[1, 2].imshow(sem_map)
    axes[1, 2].set_title('sem_map evid')
    # axes[1, 0].imshow(sem_map_mean)
    # axes[1, 0].set_title('sem_map mean')
    fig.suptitle(f'Frame {frame}')
    plt.pause(0.01)