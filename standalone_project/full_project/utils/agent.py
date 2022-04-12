import json
from os import path, symlink
import numpy as np
from utils.Tmat import TMat
from utils.bbox import Bbox2D, Bbox3D
from utils.vector import vec2, vec3, vec4
from typing import List, Tuple
import cv2 as cv
from tqdm import tqdm


import standalone_project.full_project.utils.projector as prj
import matplotlib.pyplot as plt

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

            #
            #   Get pose of the agent
            #
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


            #
            #   Get pose of the sensors
            #
            if self.label == "vehicle" or self.label == "infrastructure":
                for sens in state_json["sensors"]:
                    pose = np.array(sens['T_Mat'])
                    out = TMat()
                    out.set(pose)
                    out.handinessLeft2Right()
                    self.sensorTPoses.append(out)

            #                       
            #   Get bounding box   /!\  MUST BE RELATED WITH Agent's pose.
            #                     
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
                self.bbox3d.set_TPose(self.Tpose.get())
                 
                #   Fix the Tpose for pedestrian (they're flying)    
                if self.label == "pedestrian":
                    self.Tpose.tmat[2:3] = sz

                

        return self

        # DEBUG
        #
        # print(self.bbox3d)
        # print(self.Tpose)
        # for s in self.sensorTPoses:
        #     print(s)

    # def get_bbox_w(self):
    #     return self.Tpose * self.bbox3d

    def get_bbox3d(self) -> Bbox3D:
        return self.bbox3d

    def get_pose(self) -> TMat:
        return self.Tpose

    def get_rgb(self, frame=None) -> np.ndarray:
        if frame == None:
            return None
        img = cv.imread(f'{self.mypath}camera_rgb/{frame:06d}.png')
        return img

    def get_pred(self, frame:int) -> List[Bbox3D]:
        self.get_state(frame)
        if self.label == "pedestrian":
            raise Exception("Pedestrian do not have sensors.")
        pred_path = f"{self.mypath}/Prediction/{frame:06d}.json"
        print(self.mypath)
        # print(pred_path)
        with open(pred_path) as json_file:
            jsonData = json.load(json_file)
            # print(jsonData)
            cnt = 0
            boxes:List[Bbox3D] = []
            while True:
                try: 
                    dim = np.array(jsonData[f"{cnt}"]["dimensions (whl)"], dtype=float)
                    label = "vehicle" if jsonData[f"{cnt}"]["class"] == "car" else jsonData[f"{cnt}"]["class"]
                    Tpose = np.array(jsonData[f"{cnt}"]["K_obj2world"], dtype=float)
                    cnt += 1

                    bbox = Bbox3D(vec3(Tpose[0][3], Tpose[1][3], Tpose[2][3]), vec3(dim[0], dim[1], dim[2]), label=label)
                    bbox.set_TPose(Tpose)
                    boxes.append(bbox)
                    # print(dim)
                    # print(label)
                    # print(Tpose)
                except Exception as e:
                    break
            return boxes
        return None



    def get_visible_bbox(self, frame:int, plot:plt = None, drawBBOXonImg=False) -> Tuple[List[Bbox2D], TMat, TMat]:
        self.get_state(frame)
        if self.label == "pedestrian":
            raise Exception("Pedestrian do not have sensors.")
        kmat_path = self.mypath + "/camera_semantic_segmentation/cameraMatrix.npy"
        k_mat = prj.load_k(kmat_path)
        # np.load(kmat_path)
        # print(k_mat)
        camPose = self.sensorTPoses[0]
        with open(self.dataset_json_path) as dataset_json:
            raw_json = json.load(dataset_json)

            #every idx where idx != my id and where type is not an infrastructure
            visible_user_idx:int = [idx for idx, data in enumerate(raw_json["agents"]) if (data["type"]!="infrastructure" and idx != self.myid)]
            agents = [Agent(self.dataset_path, idx) for idx in visible_user_idx]

            img = cv.imread(self.mypath+f'camera_semantic_segmentation/{frame:06d}.png') 

            bbox2:List[Bbox2D] = []
            bbox3pts:List[List[vec2]] = []
            for a in agents:
                a.get_state(frame)
                projected = prj.projector_filter(a.get_bbox3d(), a.get_pose(), k_mat, camPose, img)
                if projected is None:
                    continue
                (bbox, points) = projected
                # print(f"\t{bbox}")
                bbox2.append(bbox)
                bbox3pts.append(points)

            # (w,h) = np.shape(img)
            h, w,_ = img.shape

            camBBox = Bbox2D(vec2(0, 0), vec2(w, h), label='terrain')
                
            bbox2 = [box for box in bbox2 if box != None]
            bbox2.insert(0, camBBox)
            # print(bbox2)
            
            img = cv.imread(self.mypath+f'camera_rgb/{frame:06d}.png')

            if drawBBOXonImg:
                color = (0, 255, 0)
                thickness = 2
                for box in bbox2:
                    points = box.get_pts()
                    pts = [tuple(np.transpose(pt.get())[0].astype(int).tolist()) for pt in points]
                    # print(pts)
                    for i in range(len(pts)):
                        img = cv.line(img, pts[i], pts[(i+1)%len(pts)], color, thickness)
                color = (0, 0, 255)
                thickness = 1
                for points in bbox3pts:
                    pts = [tuple(np.transpose(pt.get())[0].astype(int).tolist()) for pt in points]
                    # print(pts)
                    for i in range(len(pts)):
                        img = cv.line(img, pts[i], pts[(i+1)%len(pts)], color, thickness)

            if plot is not None:                    
                plot.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                plot.draw()
                plt.pause(0.001)

            return (bbox2, k_mat, camPose, self.label, cv.cvtColor(img, cv.COLOR_BGR2RGB))
                


if __name__ == "__main__":
    dataset_path:str = '/home/caillot/Documents/Dataset/CARLA_Dataset_B'
    v0 = Agent(dataset_path=dataset_path, id=18)
    print(v0)
    # v0.get_state(56)
    for i in tqdm(range(1, 100)):
        # v0.get_visible_bbox(i)
        print(v0.get_pred(i))
        
    
    # cv.waitKey(0)
    # cv.destroyAllWindows()