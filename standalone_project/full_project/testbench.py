import numpy as np
from utils.agent import Agent
from utils.vector import vec2
from utils.Tmat import TMat

from tqdm import tqdm
import json
from typing import List
import matplotlib.pyplot as plt
from cwrap.merger import mean_merger_fast, DST_merger
from cwrap.decision import cred2pign
from cwrap.metrics import toOccup
import csv
from os import path, makedirs 
from utils.metrics import create_diffmap, record, nObservMask, save_map
from utils.global_var import *
from pipeline import *
from utils.recorder import *

from agents2maps import readFE


def pipeline():
    init_out_file() # Initialize the metrics file
    (every_agents, agents2gndtruth) = read_dataset() # Read the dataset
    active_agents = get_active_agents(every_agents) # Get the active agents
    mapcenter = get_mapcenter(every_agents) # Get the map center
    for frame in tqdm(range(args.start, args.end)): # for each frame
        (mask_GND, _) = get_gnd_mask(frame, agents2gndtruth, mapcenter=mapcenter) # Get the ground truth mask
        (mask, evid_maps) = get_local_maps(frame=frame, agents=active_agents, mapcenter=mapcenter) # Get the local maps
        observed_mask = get_nObserved_mask(frame, mask) # Get the observed mask

        if CPT_MEAN: # If we want to compute the mean method
            save_map(f'{SAVE_PATH}/GND/', f'{frame:06d}.png', mask_GND, args.save_img) # Save the ground truth map (so we just need to save it once)
            mean_map = mean_merger_fast(mask, gridsize=GRIDSIZE, FE=readFE()) # Compute the mean map
            sem_map_mean = cred2pign(mean_map, method=-1) # Compute the semantic map from the mean map
            save_to_file(frame, 'avg', mask_GND, sem_map_mean, observed_mask) # Save the metrics of the mean method
            # save the maps
            if args.save_img: # If we want to save the maps
                diff = create_diffmap(mask_GND, sem_map_mean) # Compute the difference map
                save_map(f'{SAVE_PATH}/Mean/RAW/', f'{frame:06d}.png', mean_map)
                save_map(f'{SAVE_PATH}/Mean/SEM/', f'{frame:06d}.png', sem_map_mean)
                save_map(f'{SAVE_PATH}/Mean/Dif/', f'{frame:06d}.png', diff)

        evid_maps = list(evid_maps) # Convert the evidence maps to a list
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
        save_map(f'{SAVE_PATH}/{ALGO}/RAW/V-P-T/', f'{frame:06d}.png', evid_out[:,:,[1, 2, 4]], args.save_img)
        save_map(f'{SAVE_PATH}/{ALGO}/RAW/VP-VT-PT/', f'{frame:06d}.png', evid_out[:,:,[3, 5, 6]], args.save_img)
        save_map(f'{SAVE_PATH}/{ALGO}/RAW/VPT/', f'{frame:06d}.png', evid_out[:,:,7], args.save_img)

        # Test every decision taking algorithm except average max
        for decision_maker in DECIS_LUT:
            if decision_maker == 'Avg_Max':
                continue

            # fix the global evid. map to a semantic map with a given algoritm
            sem_map = cred2pign(evid_out, method=DECIS_LUT[decision_maker])
            save_to_file(frame, f'{ALGO}_{decision_maker}', mask_GND, sem_map, observed_mask)

            if args.save_img:
                diff = create_diffmap(mask_GND, sem_map)
                save_map(f'{SAVE_PATH}/{ALGO}/{decision_maker}/', f'{frame:06d}.png', sem_map)
                save_map(f'{SAVE_PATH}/{ALGO}/{decision_maker}/Dif/', f'{frame:06d}.png', diff)



if __name__ == '__main__':
    if args.gui:
        fig, axes = plt.subplots(2, 3)

    if not path.isdir(SAVE_PATH):
        makedirs(SAVE_PATH)
    pipeline()



# # FOR EACH FRAME OF A SELECTION
# for frame in tqdm(range(args.start, args.end)):

#     # Manage the GUI
#     if args.gui:
#         if CPT_MEAN:
#             axes[0, 0].imshow(mask[0])
#             axes[0, 0].set_title('Mean map')
#             axes[1, 1].imshow(mask[1])
#             axes[1, 1].set_title('Sem map mean')
#         else:
#             axes[0, 0].imshow(np.array(mask)[[0, 1, 2]].transpose(1, 2, 0))
#             axes[0, 0].set_title('Mask')
#             if type(loopback_evid) != type(None):
#                 axes[1, 1].imshow(loopback_evid[:,:,[1, 2, 4]])
#                 axes[1, 1].set_title('Loopback')
#             else:
#                 axes[1, 1].imshow(toOccup(sem_map, GRIDSIZE))
#                 axes[1, 1].set_title('Occupancy grid')
#         axes[0, 1].imshow(evid_out[:,:,[1, 2, 4]])
#         axes[0, 1].set_title('V, P, T')
#         axes[0, 2].imshow(evid_out[:,:,[3, 5, 6]])
#         axes[0, 2].set_title('VP, VT, PT')
#         axes[1, 0].imshow(mask_GND)
#         axes[1, 0].set_title('Ground truth')
#         axes[1, 2].imshow(sem_map)
#         axes[1, 2].set_title('sem_map evid')
#         fig.suptitle(f'Frame {frame}')
#         plt.pause(0.01)