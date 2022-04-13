from typing import Tuple, List
from utils.vector import vec2
from utils.Tmat import TMat
from utils.bbox import Bbox2D
import numpy as np
from utils.projector import project_BBox2DOnPlane
from utils.plucker import plkrPlane
from ctypes import *
from scipy.spatial.transform import Rotation as R
import json
from copy import deepcopy
from utils.global_var import fpSizeMax


# Evidential Grid Generator
class EGG:
    """
    Evidential Grid Generator
    """
    mapsize:int
    gridsize:int
    cellsize: float


    def __init__(self, mapsize: float, gridsize: int) -> None:
        self.mapsize = mapsize
        self.gridsize = gridsize
        self.cellsize = mapsize / float(gridsize)

    def projector_resterizer(self, agent_out:Tuple[List[Bbox2D], TMat, TMat, str], confjsonpath=None) -> List[Tuple[List[vec2], str]]:
        """
        Project the 2D bounding box on the plane

        Args:
            agent_out: the output of the agent (bbox, Tmat, Tmat, str)
            confjsonpath: the path of the configuration file

        Returns:
            a list of projected bounding box and the name of the object
        """
        (bbox_list, kmat, camT, label, _) = agent_out
        list_fp:List[Tuple[List[vec2], str]] = []
        gndPlane = plkrPlane()

        coords = []
        labels = []

        newT = camT

        # Add noise of position
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

        for bbox in bbox_list:
            # project a bbox as a footprint
            # fp = project_BBox2DOnPlane(gndPlane, bbox, kmat, camT, fpSizeMax={'vehicle': 6.00, 'pedestrian': 1.00})
            fp = project_BBox2DOnPlane(gndPlane, bbox, kmat, camT, fpSizeMax=fpSizeMax)

            # pack everything
            list_fp.extend(fp)

        return list_fp

