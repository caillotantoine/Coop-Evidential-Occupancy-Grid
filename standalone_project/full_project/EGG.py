from typing import Tuple, List
from vector import vec2
from Tmat import TMat
from bbox import Bbox2D
import numpy as np
from projector import project_BBox2DOnPlane
from plucker import plkrPlane
from ctypes import *
from scipy.spatial.transform import Rotation as R
import json
from copy import deepcopy


# Evidential Grid Generator
class EGG:
    def __init__(self, mapsize: float, gridsize: int) -> None:
        self.mapsize = mapsize
        self.gridsize = gridsize
        self.cellsize = mapsize / float(gridsize)

    def projector_resterizer(self, agent_out:Tuple[List[Bbox2D], TMat, TMat, str], confjsonpath=None):
        (bbox_list, kmat, camT, label) = agent_out
        list_fp:List[Tuple[List[vec2], str]] = []
        gndPlane = plkrPlane()

        coords = []
        labels = []

        newT = camT

        if confjsonpath != None:
            with open(confjsonpath) as json_file:
                data = json.load(json_file)
                noiseFigure = data['noise'][label]
                # print(noiseFigure)
                t = camT.get()[:3, 3]
                r = camT.get()[:3, :3]
                r = R.from_matrix(r)
                r_euler=R.as_euler(r, "xyz")
                noiseT = np.random.normal(loc=t, scale=[noiseFigure['pose_err']['x'], noiseFigure['pose_err']['y'], noiseFigure['pose_err']['z']])
                noiseR = np.random.normal(loc=r_euler, scale=[noiseFigure['rot_err']['x'], noiseFigure['rot_err']['y'], noiseFigure['rot_err']['z']])
                newT.tmat[:3, :3] = R.from_euler('xyz', noiseR).as_matrix()
                newT.tmat[:3, 3] = noiseT
        
        # print(newT)

        for bbox in bbox_list:
            fp = project_BBox2DOnPlane(gndPlane, bbox, kmat, camT)
            coords.append(np.array([(v.get().T)[0] for v in fp]))
            labels.append(bbox.label)
            list_fp.append((fp, bbox.label))

        coords = np.array(coords)
        labels = np.array([1 if l == "vehicle" else 2 for l in labels])
        # print(coords)
        # print(labels)

        return list_fp




if __name__ == "__main__":
    lib = cdll.LoadLibrary('./standalone_project/full_project/src_c/rasterizer.so')
    lib.bonjour()

