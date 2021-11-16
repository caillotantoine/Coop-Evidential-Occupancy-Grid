import json
from os import path
import numpy as np
from Tmat import TMat
from bbox import Bbox3D
from vector import vec2, vec3, vec4
from typing import List

class Agent:
    def __init__(self, dataset_path:str, id:int) -> None:
        self.dataset_path:str = dataset_path
        self.dataset_json_path:str = dataset_path + f'/information.json'
        self.pose:TMat = TMat()
        self.sensorPoses:List[TMat] = []
        self.bbox3d:Bbox3D = None

        with open(self.dataset_json_path) as dataset_json:
            info = json.load(dataset_json)
            agents = info['agents']

            self.mypath:str = self.dataset_path + '/' + agents[id]['path']
            self.label:str = agents[id]['type']

    def __str__(self):
        return f'{self.label} @ {self.mypath}'

    def get_state(self, frame:int): 
        jsonpath = self.mypath  + f'infos/{frame:06d}.json'
        print(jsonpath)
        with open(jsonpath) as f:
            state_json = json.load(f)
            if self.label == "vehicle":
                pose = np.array(state_json['vehicle']['T_Mat'])
                self.pose.set(pose)
                self.pose.handinessLeft2Right()

            elif self.label == "pedestrian":
                pose = np.array(state_json['sensors'][0]['T_Mat'])
                self.pose.set(pose)
                self.pose.handinessLeft2Right()

            elif self.label == "infrastructure":
                self.pose = None

            if self.label == "vehicle" or self.label == "infrastructure":
                for sens in state_json["sensors"]:
                    pose = np.array(sens['T_Mat'])
                    out = TMat()
                    out.set(pose)
                    out.handinessLeft2Right()
                    self.sensorPoses.append(out)


if __name__ == "__main__":
    dataset_path:str = '/home/caillot/Documents/Dataset/CARLA_Dataset_B'
    v0 = Agent(dataset_path=dataset_path, id=0)
    print(v0)
    v0.get_state(56)