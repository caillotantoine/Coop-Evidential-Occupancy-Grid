import json
from os import path, symlink
import numpy as np
from Tmat import TMat
from bbox import Bbox2D, Bbox3D
from vector import vec2, vec3, vec4
from typing import List
import cv2 as cv

import projector as prj

class Agent:
    def __init__(self, dataset_path:str, id:int) -> None:
        self.dataset_path:str = dataset_path
        self.dataset_json_path:str = dataset_path + f'/information.json'
        self.Tpose:TMat = TMat()
        self.sensorTPoses:List[TMat] = []
        self.bbox3d:Bbox3D = None
        self.myid = id

        with open(self.dataset_json_path) as dataset_json:
            info = json.load(dataset_json)
            agents = info['agents']

            self.mypath:str = self.dataset_path + '/' + agents[id]['path']
            self.label:str = agents[id]['type']

    def __str__(self):
        return f'{self.label} @ {self.mypath}'

    def get_state(self, frame:int): 
        jsonpath = self.mypath  + f'infos/{frame:06d}.json'
        # print(jsonpath)
        with open(jsonpath) as f:
            state_json = json.load(f)
            if self.label == "vehicle":
                pose = np.array(state_json['vehicle']['T_Mat'])
                self.Tpose.set(pose)
                self.Tpose.handinessLeft2Right()

            elif self.label == "pedestrian":
                pose = np.array(state_json['sensors'][0]['T_Mat'])
                self.Tpose.set(pose)
                self.Tpose.handinessLeft2Right()

            elif self.label == "infrastructure":
                self.Tpose = None

            if self.label == "vehicle" or self.label == "infrastructure":
                for sens in state_json["sensors"]:
                    pose = np.array(sens['T_Mat'])
                    out = TMat()
                    out.set(pose)
                    out.handinessLeft2Right()
                    self.sensorTPoses.append(out)

            if self.label == "vehicle" or self.label == "pedestrian":
                raw_bbox = state_json["vehicle"]["BoundingBox"]
                sx = raw_bbox["extent"]["x"]
                sy = raw_bbox["extent"]["y"]
                sz = raw_bbox["extent"]["z"]
                ox = raw_bbox["loc"]["x"] - sx
                oy = raw_bbox["loc"]["y"] - sy
                oz = raw_bbox["loc"]["z"] - sz
                bboxsize = vec3(sx*2.0, sy*2.0, sz*2.0)
                bbox_pose = vec3(ox, oy, oz)
                self.bbox3d = Bbox3D(bbox_pose, bboxsize, self.label)

        # DEBUG
        #
        # print(self.bbox3d)
        # print(self.Tpose)
        # for s in self.sensorTPoses:
        #     print(s)

    def get_bbox_w(self):
        return self.Tpose * self.bbox3d
    

    def get_visible_bbox(self, frame:int) -> List[Bbox2D]:
        self.get_state(frame)
        if self.label == "pedestrian":
            raise Exception("Pedestrian do not have sensors.")
        kmat_path = self.mypath + "/camera_semantic_segmentation/cameraMatrix.npy"
        k_mat = prj.load_k(kmat_path)
        # np.load(kmat_path)
        print(k_mat)
        camPose = self.sensorTPoses[0]
        with open(self.dataset_json_path) as dataset_json:
            raw_json = json.load(dataset_json)

            #every idx where idx != my id and where type is not an infrastructure
            visible_user_idx:int = [idx for idx, data in enumerate(raw_json["agents"]) if (data["type"]!="infrastructure" and idx != self.myid)]
            agents = [Agent(self.dataset_path, idx) for idx in visible_user_idx]
            for a in agents:
                a.get_state(frame)
                bbox3 = a.get_bbox_w()
                print(bbox3)
                bbox2 = prj.projector_filter(bbox3, k_mat, camPose, self.mypath+f'camera_semantic_segmentation/{frame:06d}.png')
                print(bbox2)

            img = cv.imread(self.mypath+f'camera_rgb/{frame:06d}.png')
            cv.imshow('image',img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            # bboxes2D = [prj.projector_filter(a.get_bbox_w(), k_mat, camPose, self.mypath+f'camera_semantic_segmentation/{frame:06d}.png') for a in agents]

            # for b in bboxes2D:
            #     print(b)
            # print(visible_user_idx)
            # print(agents)


if __name__ == "__main__":
    dataset_path:str = '/home/caillot/Documents/Dataset/CARLA_Dataset_B'
    v0 = Agent(dataset_path=dataset_path, id=18)
    print(v0)
    # v0.get_state(56)
    v0.get_visible_bbox(56)