from matplotlib.pyplot import draw
from utils.agent import Agent
from utils.global_var import dataset_path, pool
import json
from typing import List, Tuple
from agents2maps import generate_evid_grid
from utils.vector import vec2, vec3, vec4
from utils.Tmat import TMat
from utils.global_var import *
from utils.metrics import create_diffmap, record, nObservMask, save_map
import numpy as np
from utils.projector import getCwTc
from utils.Tmat import TMat
import cv2 as cv

# for parallel processing. 
# depackage the data to fit the classical argument disposition.
def get_bbox(data):
    agent, frame, drawOnImg = data
    return agent.get_visible_bbox(frame=frame, plot=None, drawBBOXonImg=drawOnImg) # Set plot to plt to show the images with the bounding boxes drawn.

def get_pred(data):
    agent, frame = data
    return agent.get_pred(frame=frame)

####
####    Processing for each agents
####
def get_bbox_par(data, parallel = False):
    if parallel:
        return pool.map(get_bbox, data)
    else:
        out = []
        for d in data:
            out.append(get_bbox(d))
        return out

def read_dataset() -> Tuple[List[Agent], List[Agent]]:
    # Read the dataset 
    # get dataset information (agents available)
    agents:List[Agent] = []
    with open(f"{dataset_path}//information.json") as json_file:
        info = json.load(json_file)
        agent_l = info['agents']
        agents = [Agent(dataset_path=dataset_path, id=idx) for idx, agent in enumerate(agent_l) if agent['type'] != "pedestrian"]
        agents2gndtruth = [Agent(dataset_path=dataset_path, id=idx) for idx, agent in enumerate(agent_l) if agent['type'] != "infrastructure"]
    return (agents, agents2gndtruth)

def get_local_maps(frame:int, agents:List[Agent], mapcenter:vec2):
    # Every agent + given frame ID
    data = [(agent, frame, True) for agent in agents]
    # from the dataset, retrieve the 2D bounding box
    bboxes = get_bbox_par(data)
    # create the evidential map + the mask for each agent
    mask_eveid_maps = [generate_evid_grid(agent_out=d, mapcenter=mapcenter) for d in bboxes]
    # datashape conversion
    # [(mask, evid_map, polygons)] -> [mask], [evid_maps]
    mask, evid_maps, polygons = zip(*mask_eveid_maps)
    #================== Start debug VY76R5FGY876T574EFU6
    #   Role of debug : print 3D bounding boxes + projected footprints in an image. 
    #                   The image will then be transfered to 
    images:List[np.ndarray] = []
    (bbox_list, kmat, camT, label, img) = zip(*bboxes)
    img = list(img)
    for idx, _ in enumerate(agents):
        sensorT = camT[idx]
        cwTc:TMat = getCwTc()
        wTc:TMat = sensorT * cwTc
        wTc.inv()   
        k = kmat[idx]

        
        color = (0, 0, 255)
        thickness = 3

        for i, polygon in enumerate(polygons[idx]):
            if i == 0:
                continue
            vertex = polygon[0]
            vertInCam = [wTc * v for v in vertex]
            projVertInImg = [k * v for v in vertInCam]
            pts_2d:List[vec2] = [pt3.nvec2() for pt3 in [pt4.vec3() for pt4 in projVertInImg]]
            imgpts = [tuple(np.transpose(pt.vec)[0].astype(int).tolist()) for pt in pts_2d]
            for i in range(len(imgpts)):
                img[idx] = cv.line(img[idx], imgpts[i], imgpts[(i+1)%len(imgpts)], color, thickness)
            
            pass


    
    # 
    #================== Start debug VY76R5FGY876T574EFU6  
    return (mask, evid_maps, img)

def get_mapcenter(agents:List[Agent]) -> vec2:
    # Print aquired active agents & compute scene center from infrastructures' poses
    iPose:List[TMat] = []
    for idx, agent in enumerate(agents):
        print(f"{idx} : \t{agent}")
        if agent.label == "infrastructure":
            p = agent.get_state(args.start).sensorTPoses[0].get_translation()
            iPose.append(p.vec3().vec2())
    sum:vec2 = vec2(0.0, 0.0)
    if len(iPose) == 0:
        raise ValueError(f"No infrastructure found in the dataset.")
    for p in iPose:
        sum += p
    scene_center = sum / len(iPose)
    return scene_center

def get_active_agents(agents:List[Agent]) -> List[Agent]:
    # Get the "active" agent
    try:
        with open(args.json_path) as json_file:
            data = json.load(json_file)
            # Active agents are selected here
            idx2read = data['index to read'] # [1, 6, 7, 8, 9, 10, 11]
            json_file.close()
            if idx2read != None:
                agents = [agents[i] for i in idx2read]
            return agents
    except:
        raise ValueError(f"Could not read the json file {args.json_path}")

def get_nObserved_mask(frame:int, mask:List[np.ndarray]) -> np.ndarray:
    # get to know which cells are observed
    observed_zones = nObservMask(mask)
    save_map(f'{SAVE_PATH}/CoopZone/', f'{frame:06d}.png', observed_zones, args.save_img)
    return observed_zones

def get_gnd_mask(frame:int, agents2gndtruth:List[Agent], mapcenter:vec2) -> Tuple[np.ndarray, np.ndarray]:
    # with every agent info, create the ground truth map
    gnd_agent = [agent.get_state(frame).get_bbox3d() for agent in agents2gndtruth]
    mask_eveid_maps_GND = generate_evid_grid(agent_3D=gnd_agent, mapcenter=mapcenter)
    # (mask_GND, evid_maps_GND) = mask_eveid_maps_GND
    return mask_eveid_maps_GND
