from  utils.global_var import *
from utils.Tmat import TMat
from standalone_project.full_project.utils.EGG import EGG
from typing import List, Tuple
from utils.bbox import Bbox2D, Bbox3D
from utils.vector import vec2, vec4
import numpy as np
import cwrap.rasterizer as rasterizer
import cv2 as cv
import json

def readFE() -> List[List[float]]:
    try:
        with open(args.json_path) as json_file:
            data = json.load(json_file)
            FE = data['FE_mat']
            json_file.close()
            return FE
    except:
        raise("Error: No json file found")

# Generate the local evidential map from one agent
def generate_evid_grid(agent_out:Tuple[List[Bbox2D], TMat, TMat, str] = None, mapcenter:vec2 = vec2(x=0.0, y=0.0), agent_3D:List[Bbox3D] = None, antoine=False):

    egg = EGG(mapsize=MAPSIZE, gridsize=(GRIDSIZE)) # create an Evidential Grid Generator
    mask = np.zeros(shape=(GRIDSIZE, GRIDSIZE), dtype=np.uint8) # empty mask map

    polygons:List[Tuple[np.ndarray, str]] = []

    if agent_out != None:
        # Extract 2D footprints and the label from 2d bounding box of the dataset
        eggout = egg.projector_resterizer(agent_out, confjsonpath=args.json_path)

        # Extract the 2D footprints 
        fp_poly = np.array([np.array([(v.get().T)[0] for v in poly], dtype=np.float32) for (poly, _) in eggout])

        # Extract the labels
        fp_label = np.array([1 if label == 'vehicle' else 2 if label == 'pedestrian' else 3 if label == 'terrain' else 0 for (_, label) in eggout], dtype=
        np.int32)

        polygons = [([vec4(x=v[0], y=v[1], z=0) for v in fp_poly[i]], label) for i, label in enumerate(list(fp_label))] 



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
    FE = readFE()

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
    return (mask, evid_map, polygons)