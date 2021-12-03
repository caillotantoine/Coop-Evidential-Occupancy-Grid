from typing import Tuple, List
from vector import vec2
from Tmat import TMat
from bbox import Bbox2D
import numpy as np
from projector import project_BBox2DOnPlane
from plucker import plkrPlane
from ctypes import *

# Evidential Grid Generator
class EGG:
    def __init__(self, mapsize: float, gridsize: int) -> None:
        self.mapsize = mapsize
        self.gridsize = gridsize
        self.cellsize = mapsize / float(gridsize)

    def projector_resterizer(self, agent_out:Tuple[List[Bbox2D], TMat, TMat]):
        (bbox_list, kmat, camT) = agent_out
        list_fp:List[Tuple[List[vec2], str]] = []
        gndPlane = plkrPlane()

        coords = []
        labels = []

        for bbox in bbox_list:
            fp = project_BBox2DOnPlane(gndPlane, bbox, kmat, camT)
            coords.append(np.array([(v.get().T)[0] for v in fp]))
            labels.append(bbox.label)
            list_fp.append((fp, bbox.label))

        coords = np.array(coords)
        labels = np.array([1 if l == "vehicle" else 2 for l in labels])


        print(coords)
        print(labels)
        

        

        return list_fp




if __name__ == "__main__":
    lib = cdll.LoadLibrary('./standalone_project/full_project/src_c/rasterizer.so')
    lib.bonjour()

