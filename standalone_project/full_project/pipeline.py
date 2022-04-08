from utils.agent import Agent
from utils.global_var import dataset_path, pool
import json
from typing import List, Tuple
from agents2maps import generate_evid_grid
from utils.vector import vec2
from utils.Tmat import TMat
from utils.global_var import *
from utils.metrics import create_diffmap, record, nObservMask, save_map
import numpy as np

# for parallel processing. 
# depackage the data to fit the classical argument disposition.
def get_bbox(data):
    agent, frame = data
    return agent.get_visible_bbox(frame=frame, plot=None) # Set plot to plt to show the images with the bounding boxes drawn.

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
    data = [(agent, frame) for agent in agents]
    # from the dataset, retrieve the 2D bounding box
    bboxes = get_bbox_par(data)
    # create the evidential map + the mask for each agent
    mask_eveid_maps = [generate_evid_grid(agent_out=d, mapcenter=mapcenter) for d in bboxes]
    # datashape conversion
    # [(mask, evid_map)] -> [mask], [evid_maps]
    mask, evid_maps = zip(*mask_eveid_maps)
    return (mask, evid_maps)

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
