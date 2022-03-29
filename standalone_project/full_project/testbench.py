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
from metrics import TFPN, toOccup
import csv
import sys
import argparse
from os import path, makedirs 

import rasterizer

# Initial map descriptors
MAPSIZE = 120.0
STEPGRID = 5
GRIDSIZE = int(MAPSIZE) * STEPGRID

# Look up tables
FUS_LUT = {'Dempster': 0, 'Conjunctive': 1, 'Disjunctive': 2}
LABEL_LUT = {'Vehicle': int('0b01000000', 2), 'Pedestrian': int('0b10000000', 2), 'Terrain': int('0b00000010', 2)}
DECIS_LUT = {'Avg_Max': -1, 'BetP': 0, 'Bel': 1, 'Pl': 2, 'BBA': 3}
TFPN_LUT = {'TP': 0, 'TN': 1, 'FP': 2, 'FN': 3}


# Manage the arguments
argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
    '--algo',
    metavar='A',
    default='Dempster',
    help='Choose between Dempster, Conjunctive and Disjunctive (default Dempster).')
argparser.add_argument(
    '--mean',
    metavar='M',
    type=bool,
    default=False,
    help='Compute the mean (default False).')
argparser.add_argument(
    '--gui',
    metavar='G',
    type=bool,
    default=False,
    help='Show the GUI (default False).')
argparser.add_argument(
    '--save_img',
    type=bool,
    default=False,
    help='Save maps as images (default False).')
argparser.add_argument(
    '--start',
    metavar='S',
    type=int,
    default=10,
    help='Starting point in the dataset (default 10).')
argparser.add_argument(
    '--end',
    metavar='E',
    type=int,
    default=500,
    help='Ending point in the dataset (default 500).')
argparser.add_argument(
    '--dataset_path',
    default='/home/caillot/Documents/Dataset/CARLA_Dataset_B',
    help='Path of the dataset.')
argparser.add_argument(
    '--save_path',
    default='/home/caillot/Documents/output_algo/',
    help='Saving path.')

argparser.add_argument(
    '--json_path',
    default='./standalone_project/full_project/configs/config_perfect_full.json',
    help='Configuration json file path.')

args = argparser.parse_args()
print(args)

SAVE_PATH = args.save_path
CPT_MEAN = args.mean
ALGO = args.algo
ALGOID = FUS_LUT[ALGO]
dataset_path:str = args.dataset_path

if args.gui:
    fig, axes = plt.subplots(2, 3)

if not path.isdir(SAVE_PATH):
    makedirs(SAVE_PATH)

ANTOINE_M = False

# for parallel processing. 
# depackage the data to fit the classical argument disposition.
def get_bbox(data):
    agent, frame = data
    return agent.get_visible_bbox(frame=frame)

def get_pred(data):
    agent, frame = data
    return agent.get_pred(frame=frame)



# Generate the local evidential map from one agent
def generate_evid_grid(agent_out:Tuple[List[Bbox2D], TMat, TMat, str] = None, agent_3D:List[Bbox3D] = None, antoine=False):

    egg = EGG(mapsize=MAPSIZE, gridsize=(GRIDSIZE)) # create an Evidential Grid Generator
    mask = np.zeros(shape=(GRIDSIZE, GRIDSIZE), dtype=np.uint8) # empty mask map

    if agent_out != None:
        # Extract 2D footprints and the label from 2d bounding box of the dataset
        eggout = egg.projector_resterizer(agent_out, confjsonpath=args.json_path)

        # Extract the 2D footprints 
        fp_poly = np.array([np.array([(v.get().T)[0] for v in poly], dtype=np.float32) for (poly, _) in eggout])

        # Extract the labels
        fp_label = np.array([1 if label == 'vehicle' else 2 if label == 'pedestrian' else 3 if label == 'terrain' else 0 for (_, label) in eggout], dtype=
        np.int32)

        # Rasterize the 2D footprints and create a mask
        rasterizer.projector(len(fp_label), fp_label, fp_poly, mask, MAPSIZE, GRIDSIZE)


    elif agent_3D != None:
        # TO DO : Treating 3D bounding box
        for agent in agent_3D:
            bboxsize = agent.get_size()
            label = agent.get_label()
            poseT = agent.get_TPose()
            
            bin_mask = [np.array([0.5, 0.5, 0.0, 1.0]), np.array([0.5, -0.5, 0.0, 1.0]), np.array([-0.5, -0.5, 0.0, 1.0]), np.array([-0.5, 0.5, 0.0, 1.0])]
            
            centered_fps = [(bboxsize.vec4().get().T * m).T for m in bin_mask]

            fps = [poseT @ v for v in centered_fps]
            if antoine:
                fps_pix = np.array([np.array((((v * STEPGRID) + (GRIDSIZE / 2)).T)[0][[0, 2]], dtype=int) for v in fps])
            else:
                fps_pix = np.array([np.array((((v * STEPGRID) + (GRIDSIZE / 2)).T)[0][:2], dtype=int) for v in fps])
            
            
            cv.fillPoly(mask, pts=[fps_pix], color=(int('0b01000000', 2) if label == 'vehicle' else int('0b10000000', 2)))

    else: 
        raise NameError("Both agent_out and agent_3D are set to None. Assign a value to at least one of them.")
        
    # define the number of focal elements
    nFE = 8
    # grab the focal elements from a json file
    with open(args.json_path) as json_file:
        data = json.load(json_file)
        FE = data['FE_mat']
        json_file.close()

    # example of focal elements
    # #      Ø    V    P    VP   T    VT   PT   VPT
    # FE = [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9], # VPT
    #     [0.1, 0.6, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1], # V
    #     [0.1, 0.0, 0.6, 0.1, 0.0, 0.0, 0.1, 0.1], # P
    #     [0.1, 0.0, 0.0, 0.0, 0.6, 0.1, 0.1, 0.1]] # T

    # convert the focal element list as nparray
    FE = np.array(FE, dtype=np.float32)  

    # create an empty evidential map with the right size   
    evid_map = np.zeros(shape=(GRIDSIZE, GRIDSIZE, nFE), dtype=np.float32)

    # from the masks, create evidential maps
    rasterizer.apply_BBA(nFE, GRIDSIZE, FE, mask, evid_map)

    # return the masks (for averaging method) and the evidential maps
    return (mask, evid_map)



# get the required value to compute the metrics
def record(sem_gnd:np.ndarray, sem_test:np.ndarray, gridsize:int, frame:int):
    occ_gnd = toOccup(sem_gnd, gridsize)
    occ_test = toOccup(sem_test, gridsize)

    outs = {'frame': frame}
    for key in TFPN_LUT:
        outs[f'occup_{key}'] = TFPN(occ_gnd, occ_test, gridsize, TFPN_LUT[key], 0)

    for keylab in LABEL_LUT:
        for key_tfpn in TFPN_LUT:
            outs[f'{keylab}_{key_tfpn}'] = TFPN(sem_gnd, sem_test, gridsize, TFPN_LUT[key_tfpn], LABEL_LUT[keylab])

    return outs


########################################
###                                  ###
###          STARTING POINT          ###
###                                  ###
########################################

# Read the dataset 
# get dataset information (agents available)
agents:List[Agent] = []
with open(f"{dataset_path}//information.json") as json_file:
    info = json.load(json_file)
    agent_l = info['agents']
    agents = [Agent(dataset_path=dataset_path, id=idx) for idx, agent in enumerate(agent_l) if agent['type'] != "pedestrian"]
    agents2gndtruth = [Agent(dataset_path=dataset_path, id=idx) for idx, agent in enumerate(agent_l) if agent['type'] != "infrastructure"]

# Get the "active" agent
with open(args.json_path) as json_file:
    data = json.load(json_file)
    # Active agents are selected here
    idx2read = data['index to read'] # [1, 6, 7, 8, 9, 10, 11]
    json_file.close()
    if idx2read != None:
        agents = [agents[i] for i in idx2read]

# Print aquired active agents
for idx, agent in enumerate(agents):
    print(f"{idx} : \t{agent}")

# Prepare metrics recordings 
fieldsname = ['frame']
for key in TFPN_LUT:
    fieldsname.append(f'occup_{key}')

for keylab in LABEL_LUT:
    for key_tfpn in TFPN_LUT:
        fieldsname.append(f'{keylab}_{key_tfpn}')

# print(fieldsname)

# create the outputs file for avg method
recfile = open(f'{SAVE_PATH}/avg.csv', mode='w')
writer = csv.DictWriter(recfile, fieldnames=fieldsname)
writer.writeheader()
recfile.close()

# create the outputs file for other methods
for decision_maker in DECIS_LUT:
    if decision_maker == 'Avg_Max':
        continue
    recfile = open(f'{SAVE_PATH}/{ALGO}_{decision_maker}.csv', mode='w')
    writer = csv.DictWriter(recfile, fieldnames=fieldsname)
    writer.writeheader()
    recfile.close()

####
####    Processing for each frames
####

# // processing setup
pool = Pool(multiprocessing.cpu_count())

# FOR EACH FRAME OF A SELECTION
for frame in tqdm(range(args.start, args.end)):

    # Every agent + given frame ID
    data = [(agent, frame) for agent in agents]
    if ANTOINE_M:
        # TODO
        pass
    else:
        # from the dataset, retrieve the 2D bounding box
        bboxes = pool.map(get_bbox, data)
        # create the evidential map + the mask for each agent
        mask_eveid_maps = [generate_evid_grid(agent_out=d) for d in bboxes]
    # datashape conversion
    # [(mask, evid_map)] -> [mask], [evid_maps]
    mask, evid_maps = zip(*mask_eveid_maps)
    
    # with every agent info, create the ground truth map
    gnd_agent = [agent.get_state(frame).get_bbox3d() for agent in agents2gndtruth]
    mask_eveid_maps_GND = generate_evid_grid(agent_3D=gnd_agent)
    (mask_GND, evid_maps_GND) = mask_eveid_maps_GND

    

    # Merging with memory method
    if CPT_MEAN:

        # Not related but save the ground truth.
        # Since average is calculated once, ground truth is saved once to speed up
        # TODO : create a dedicated argument
        if args.save_img:
            if not path.isdir(f'{SAVE_PATH}/GND/'):
                makedirs(f'{SAVE_PATH}/GND/')    
            plt.imsave(f'{SAVE_PATH}/GND/{frame:06d}.png', mask_GND)


        # merge the maps with joint probability
        mean_map = mean_merger_fast(mask, gridsize=GRIDSIZE)
        # take the decision from probability to fixed semantic
        sem_map_mean = cred2pign(mean_map, method=-1)
        # Save the value required for the metrics
        with open(f'{SAVE_PATH}/avg.csv', mode='a') as recfile:
            writer = csv.DictWriter(recfile, fieldnames=fieldsname)
            writer.writerow(record(mask_GND, sem_map_mean, GRIDSIZE, frame))
            recfile.close()
        
        # save the maps
        if args.save_img:
            if not path.isdir(f'{SAVE_PATH}/Mean/RAW/'):
                makedirs(f'{SAVE_PATH}/Mean/RAW/')
            if not path.isdir(f'{SAVE_PATH}/Mean/SEM/'):
                makedirs(f'{SAVE_PATH}/Mean/SEM/')    
            maprec = record(mask_GND, sem_map_mean, GRIDSIZE, frame)
            plt.imsave(f'{SAVE_PATH}/Mean/RAW/{frame:06d}.png', mean_map)
            plt.imsave(f'{SAVE_PATH}/Mean/SEM/{frame:06d}.png', sem_map_mean)

    # Merge the evidential map for a given algorithm 
    evid_out = DST_merger(evid_maps=list(evid_maps), gridsize=GRIDSIZE, CUDA=False, method=ALGOID)
    # Save the maps
    if args.save_img:
        if not path.isdir(f'{SAVE_PATH}/{ALGO}/RAW/'):
            makedirs(f'{SAVE_PATH}/{ALGO}/RAW/')
        plt.imsave(f'{SAVE_PATH}/{ALGO}/RAW/{frame:06d}-v-p-t.png', evid_out[:,:,[1, 2, 4]])
        plt.imsave(f'{SAVE_PATH}/{ALGO}/RAW/{frame:06d}-vp-vt-pt.png', evid_out[:,:,[3, 5, 6]])
        plt.imsave(f'{SAVE_PATH}/{ALGO}/RAW/{frame:06d}-vpt.png', evid_out[:,:,7])

    # Test every decision taking algorithm except average max
    for decision_maker in DECIS_LUT:
        if decision_maker == 'Avg_Max':
            continue
        
        # fix the global evid. map to a semantic map with a given algoritm
        sem_map = cred2pign(evid_out, method=DECIS_LUT[decision_maker])
        with open(f'{SAVE_PATH}/{ALGO}_{decision_maker}.csv', mode='a') as recfile:
            writer = csv.DictWriter(recfile, fieldnames=fieldsname)
            writer.writerow(record(mask_GND, sem_map, GRIDSIZE, frame))
            recfile.close()

        # Save the semantic map
        if args.save_img:
            if not path.isdir(f'{SAVE_PATH}/{ALGO}/{decision_maker}/'):
                makedirs(f'{SAVE_PATH}/{ALGO}/{decision_maker}/')
            plt.imsave(f'{SAVE_PATH}/{ALGO}/{decision_maker}/{frame:06d}.png', sem_map)

    # Manage the GUI
    if args.gui:
        if CPT_MEAN:
            axes[0, 0].imshow(mean_map)
            axes[0, 0].set_title('Mean map')
            axes[1, 1].imshow(sem_map_mean)
            axes[1, 1].set_title('Sem map mean')
        else:
            axes[0, 0].imshow(np.array(mask)[[0, 1, 2]].transpose(1, 2, 0))
            axes[0, 0].set_title('Mask')
            axes[1, 1].imshow(toOccup(sem_map, GRIDSIZE))
            axes[1, 1].set_title('Occupancy grid')
        axes[0, 1].imshow(evid_out[:,:,[1, 2, 4]])
        axes[0, 1].set_title('V, P, T')
        axes[0, 2].imshow(evid_out[:,:,[3, 5, 6]])
        axes[0, 2].set_title('VP, VT, PT')
        axes[1, 0].imshow(mask_GND)
        axes[1, 0].set_title('Ground truth')
        axes[1, 2].imshow(sem_map)
        axes[1, 2].set_title('sem_map evid')
        fig.suptitle(f'Frame {frame}')
        plt.pause(0.01)