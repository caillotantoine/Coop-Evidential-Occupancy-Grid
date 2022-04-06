from copy import deepcopy
import ctypes
import multiprocessing
from turtle import color
from joblib import parallel_backend
import matplotlib
import numpy as np
from torch import dtype
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
    default=True,
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
    default=True,
    help='Save maps as images (default False).')
argparser.add_argument(
    '--loopback_evid',
    type=bool,
    default=False,
    help='Loop back the t-1 evidential map as an entry of the agents (default False).')
argparser.add_argument(
    '--start',
    metavar='S',
    type=int,
    default=130, #10
    help='Starting point in the dataset (default 10).')
argparser.add_argument(
    '--pdilate',
    type=int,
    default=-1, #10
    help='Pedestrian Dilation Factor. -1: Off, Choose a value between 0 and 5. (default -1)')
argparser.add_argument(
    '--cooplvl',
    type=int,
    default=2, #10
    help='Number of observation to be a valid measure. -1: All, Choose a value between 0 and N users. (default 2)')
argparser.add_argument(
    '--gdilate',
    type=int,
    default=-1, #10
    help='Dilation Factor for every object at mask level. -1: Off, Choose a value between 0 and 5. (default -1)')
argparser.add_argument(
    '--end',
    metavar='E',
    type=int,
    default=160,
    help='Ending point in the dataset (default 500).')
argparser.add_argument(
    '--dataset_path',
    default='/home/caillot/Documents/Dataset/CARLA_Dataset_intersec_dense',
    help='Path of the dataset.')
argparser.add_argument(
    '--save_path',
    default='/home/caillot/Documents/output_algo/',
    help='Saving path.')

argparser.add_argument(
    '--json_path',
    default='./standalone_project/full_project/configs/config_perfect_full_testBBA15.json',
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
    return agent.get_visible_bbox(frame=frame, plot=None) # Set plot to plt to show the images with the bounding boxes drawn.

def get_pred(data):
    agent, frame = data
    return agent.get_pred(frame=frame)



# Generate the local evidential map from one agent
def generate_evid_grid(agent_out:Tuple[List[Bbox2D], TMat, TMat, str] = None, mapcenter:vec2 = vec2(x=0.0, y=0.0), agent_3D:List[Bbox3D] = None, antoine=False):

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
        rasterizer.projector(len(fp_label), fp_label, fp_poly, mask, mapcenter, MAPSIZE, GRIDSIZE)

        # Dilate everything on teh mask
        if args.gdilate >= 0:
            dilatation_size = args.gdilate
            element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
            for o in [64, 128]:
                obj_mask = np.bitwise_and(mask, o, dtype=np.uint8)
                dil_ped_mask = cv.dilate(obj_mask, element)
                mask = np.where(dil_ped_mask == o, dil_ped_mask, mask)


    elif agent_3D != None:
        # TO DO : Treating 3D bounding box
        mask = np.where(mask == 0, 2, mask)
        for agent in agent_3D:
            bboxsize = agent.get_size()
            label = agent.get_label()
            poseT = agent.get_TPose()
            
            bin_mask = [np.array([0.5, 0.5, 0.0, 1.0]), np.array([0.5, -0.5, 0.0, 1.0]), np.array([-0.5, -0.5, 0.0, 1.0]), np.array([-0.5, 0.5, 0.0, 1.0])]
            
            centered_fps = [(bboxsize.vec4().get().T * m).T for m in bin_mask]

            fps = [poseT @ v for v in centered_fps]
            fps2 = [np.array([[v[0]-mapcenter.x(), v[1]-mapcenter.y(), 0.0, 1.0]]).T for v in fps]
            if antoine:
                fps_pix = np.array([np.array((((v * STEPGRID) + (GRIDSIZE / 2)).T)[0][[0, 2]], dtype=int) for v in fps2])
            else:
                fps_pix = np.array([np.array((((v * STEPGRID) + (GRIDSIZE / 2)).T)[0][:2], dtype=int) for v in fps2])
            
            
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


def IoU(TP:int, FP:int, FN:int) -> float:
    try:
        return (float(TP) / float(TP + FP + FN)) * 100.0
    except ZeroDivisionError:
        return np.NaN

def F1(TP:int, FP:int, FN:int) -> float:
    try:
        return (float(TP) / (float(TP) + (float(FP + FN) / 2.0))) * 100.0
    except ZeroDivisionError:
        return np.NaN

def CR(TP:int, TN:int, FP:int, FN:int) -> float:
    try:
        return (float(TP+TN) / float(TP+TN+FP+FN)) * 100.0
    except ZeroDivisionError:
        return np.NaN

# get the required value to compute the metrics
def record(sem_gnd:np.ndarray, sem_test:np.ndarray, zones:np.ndarray, coop_lvl:int, gridsize:int, frame:int):
    occ_gnd = toOccup(sem_gnd, gridsize)
    occ_test = toOccup(sem_test, gridsize)

    outs = {'frame': frame}
    for key in TFPN_LUT:
        outs[f'occup_{key}'] = TFPN(occ_gnd, occ_test, zones, coop_lvl, gridsize, TFPN_LUT[key], 0)
    outs[f'occup_IoU'] = IoU(outs[f'occup_TP'], outs[f'occup_FP'], outs[f'occup_FN'])
    outs[f'occup_F1'] = F1(outs[f'occup_TP'], outs[f'occup_FP'], outs[f'occup_FN'])
    outs[f'occup_CR'] = CR(outs[f'occup_TP'], outs[f'occup_TN'], outs[f'occup_FP'], outs[f'occup_FN'])

    mIoU = 0.0
    mF1 = 0.0
    for keylab in LABEL_LUT:
        for key_tfpn in TFPN_LUT:
            outs[f'{keylab}_{key_tfpn}'] = TFPN(sem_gnd, sem_test, zones, coop_lvl, gridsize, TFPN_LUT[key_tfpn], LABEL_LUT[keylab])
        
        outs[f'{keylab}_IoU'] = IoU(outs[f'{keylab}_TP'], outs[f'{keylab}_FP'], outs[f'{keylab}_FN'])
        outs[f'{keylab}_F1'] = F1(outs[f'{keylab}_TP'], outs[f'{keylab}_FP'], outs[f'{keylab}_FN'])
        outs[f'{keylab}_CR'] = CR(outs[f'{keylab}_TP'], outs[f'{keylab}_TN'], outs[f'{keylab}_FP'], outs[f'{keylab}_FN'])
        mIoU += outs[f'{keylab}_IoU']
        mF1 += outs[f'{keylab}_F1']
    outs[f'mIoU'] = mIoU / len(LABEL_LUT)
    outs[f'mF1'] = mF1 / len(LABEL_LUT)

    return outs


# give the numlber of observation per cells
def nObservMask(masks_in:List[np.ndarray]) -> np.ndarray:
    maskout = np.zeros(shape=masks_in[0].shape)
    for mask in masks_in:
        maskout += np.where(mask > 0, 1, 0)
    return maskout.astype(np.uint8)

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


# Print aquired active agents & compute scene center from infrastructures' poses
iPose:List[TMat] = []
for idx, agent in enumerate(agents):
    print(f"{idx} : \t{agent}")
    if agent.label == "infrastructure":
        p = agent.get_state(args.start).sensorTPoses[0].get_translation()
        print(p)
        print(p.vec3().vec2())
        iPose.append(p.vec3().vec2())

sum:vec2 = vec2(0.0, 0.0)
for p in iPose:
    sum += p
scene_center = sum / len(iPose)
print(f"Scene center : {scene_center}")


# Prepare metrics recordings 
fieldsname = ['frame', 'mIoU', 'mF1', 'occup_IoU', 'occup_F1', 'Vehicle_IoU', 'Terrain_F1', 'Vehicle_F1', 'Pedestrian_IoU', 'Terrain_IoU', 'Pedestrian_F1', 'Terrain_CR', 'Vehicle_CR', 'occup_CR', 'Pedestrian_CR']
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

def get_bbox_par(data, parallel = False):
    if parallel:
        return pool.map(get_bbox, data)
    else:
        out = []
        for d in data:
            out.append(get_bbox(d))
        return out


# Loop back of the evidential map
loopback_evid = None

# FOR EACH FRAME OF A SELECTION
for frame in tqdm(range(args.start, args.end)):

    # Every agent + given frame ID
    data = [(agent, frame) for agent in agents]
    if ANTOINE_M:
        # TODO
        pass
    else:
        # from the dataset, retrieve the 2D bounding box
        bboxes = get_bbox_par(data)
        # create the evidential map + the mask for each agent
        mask_eveid_maps = [generate_evid_grid(agent_out=d, mapcenter=scene_center) for d in bboxes]
    # datashape conversion
    # [(mask, evid_map)] -> [mask], [evid_maps]
    mask, evid_maps = zip(*mask_eveid_maps)
    
    

    # get to know which cells are observed
    observed_zones = nObservMask(mask)
    if args.save_img:
        if not path.isdir(f'{SAVE_PATH}/CoopZone/'):
            makedirs(f'{SAVE_PATH}/CoopZone/')    
        plt.imsave(f'{SAVE_PATH}/CoopZone/{frame:06d}.png', observed_zones)

    
    
    # with every agent info, create the ground truth map
    gnd_agent = [agent.get_state(frame).get_bbox3d() for agent in agents2gndtruth]
    mask_eveid_maps_GND = generate_evid_grid(agent_3D=gnd_agent, mapcenter=scene_center)
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
        with open(args.json_path) as json_file:
            data = json.load(json_file)
            FE = data['FE_mat']
            json_file.close()
            mean_map = mean_merger_fast(mask, gridsize=GRIDSIZE, FE=FE)
            # take the decision from probability to fixed semantic
        sem_map_mean = cred2pign(mean_map, method=-1)
        # Save the value required for the metrics
        with open(f'{SAVE_PATH}/avg.csv', mode='a') as recfile:
            writer = csv.DictWriter(recfile, fieldnames=fieldsname)
            writer.writerow(record(mask_GND, sem_map_mean, observed_zones, args.cooplvl, GRIDSIZE, frame))
            recfile.close()
        
        # save the maps
        if args.save_img:
            if not path.isdir(f'{SAVE_PATH}/Mean/RAW/'):
                makedirs(f'{SAVE_PATH}/Mean/RAW/')
            if not path.isdir(f'{SAVE_PATH}/Mean/SEM/'):
                makedirs(f'{SAVE_PATH}/Mean/SEM/')    
            maprec = record(mask_GND, sem_map_mean, observed_zones, args.cooplvl, GRIDSIZE, frame)
            plt.imsave(f'{SAVE_PATH}/Mean/RAW/{frame:06d}.png', mean_map)
            plt.imsave(f'{SAVE_PATH}/Mean/SEM/{frame:06d}.png', sem_map_mean)
            diff = np.zeros((GRIDSIZE, GRIDSIZE, 3), dtype=np.float)
            diff[:, :, 0] = mask_GND
            diff[:, :, 2] = sem_map_mean
            # diff[:, :, 2] = np.absolute(mask_GND-sem_map_mean)
            diff /= 255.0
            if not path.isdir(f'{SAVE_PATH}/Mean/Dif/'):
                makedirs(f'{SAVE_PATH}/Mean/Dif/')    
            plt.imsave(f'{SAVE_PATH}/Mean/Dif/{frame:06d}.png', diff)

    evid_maps = list(evid_maps)

    if args.loopback_evid:
        evid_buffer = DST_merger(evid_maps=evid_maps, gridsize=GRIDSIZE, CUDA=False, method=ALGOID)
        # insert the loopback as first element
        if type(loopback_evid) != type(None):
            evid_out = DST_merger(evid_maps=[loopback_evid, evid_buffer], gridsize=GRIDSIZE, CUDA=False, method=ALGOID)
        else:
            evid_out = evid_buffer
        loopback_evid = evid_buffer
    else:
        # Merge the evidential map for a given algorithm
        evid_out = DST_merger(evid_maps=evid_maps, gridsize=GRIDSIZE, CUDA=False, method=ALGOID)


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


        # Grow pedestrians
        if args.pdilate >= 0:
            pedestrian_mask = np.bitwise_and(sem_map, 128, dtype=np.uint8)
            dilatation_size = args.pdilate 
            element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
            dil_ped_mask = cv.dilate(pedestrian_mask, element)
            sem_map = np.where(dil_ped_mask == 128, dil_ped_mask, sem_map)

        with open(f'{SAVE_PATH}/{ALGO}_{decision_maker}.csv', mode='a') as recfile:
            writer = csv.DictWriter(recfile, fieldnames=fieldsname)
            writer.writerow(record(mask_GND, sem_map, observed_zones, args.cooplvl, GRIDSIZE, frame))
            recfile.close()

        # Save the semantic map
        if args.save_img:
            if not path.isdir(f'{SAVE_PATH}/{ALGO}/{decision_maker}/'):
                makedirs(f'{SAVE_PATH}/{ALGO}/{decision_maker}/')
            plt.imsave(f'{SAVE_PATH}/{ALGO}/{decision_maker}/{frame:06d}.png', sem_map)
            diff = np.zeros((GRIDSIZE, GRIDSIZE, 3), dtype=np.float)
            diff[:, :, 0] = mask_GND
            diff[:, :, 2] = sem_map
            # diff[:, :, 2] = np.absolute(mask_GND-sem_map)
            diff /= 255.0
            if not path.isdir(f'{SAVE_PATH}/{ALGO}/Dif/'):
                makedirs(f'{SAVE_PATH}/{ALGO}/Dif/')    
            plt.imsave(f'{SAVE_PATH}/{ALGO}/Dif/{frame:06d}.png', diff)


    # Manage the GUI
    if args.gui:
        if CPT_MEAN:
            axes[0, 0].imshow(mask[0])
            axes[0, 0].set_title('Mean map')
            axes[1, 1].imshow(mask[1])
            axes[1, 1].set_title('Sem map mean')
        else:
            axes[0, 0].imshow(np.array(mask)[[0, 1, 2]].transpose(1, 2, 0))
            axes[0, 0].set_title('Mask')
            if type(loopback_evid) != type(None):
                axes[1, 1].imshow(loopback_evid[:,:,[1, 2, 4]])
                axes[1, 1].set_title('Loopback')
            else:
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