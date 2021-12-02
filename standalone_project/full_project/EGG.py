from typing import Tuple, List
from vector import vec2
from Tmat import TMat
from bbox import Bbox2D
import numpy as np
from projector import project_BBox2DOnPlane
from plucker import plkrPlane


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

        for bbox in bbox_list:
            fp = project_BBox2DOnPlane(gndPlane, bbox, kmat, camT)
            list_fp.append((fp, bbox.label))
        
        return list_fp




