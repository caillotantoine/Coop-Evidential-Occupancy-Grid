#!/usr/bin/env python3
# coding: utf-8


# /!\ THIS FILE IS NOT A ROS NODE

from os import path
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import utils
import cv2 as cv
import matplotlib.pyplot as plt

class BBox3DExtractor:
    def __init__(self, json_path):
        self.pts = []
        if path.exists(json_path):
            data = None
            with open(json_path) as f:
                    data = json.load(f)
            if data != None:
                ox = data['vehicle']['BoundingBox']['loc']['x']
                oy = data['vehicle']['BoundingBox']['loc']['y']
                oz = data['vehicle']['BoundingBox']['loc']['z']
                l = data['vehicle']['BoundingBox']['extent']['x']
                w = data['vehicle']['BoundingBox']['extent']['y']
                h = data['vehicle']['BoundingBox']['extent']['z']
                oT = np.identity(4)
                oT[0,3] = ox
                oT[1,3] = oy
                oT[2,3] = oz
                s = np.array([l, w, h])
                vT = utils.changeHand(np.array(data['vehicle']['T_Mat']))
                

                for i in [0,1,3,2, 6, 7, 5, 4]:
                    mask = np.array(list(map(int,list("{0:03b}".format(i)))))
                    mask = np.where(mask==1,1,-1)
                    vec = s * mask
                    vecout = np.zeros(4)
                    vecout[:3] = vec
                    vecout[3] = 1

                    vecout = np.matmul(oT, vecout)
                    pt = np.matmul(vT, vecout)
                    self.pts.append(pt)
            else:
                raise NameError('No bounding box info in %s'%path)
        else:
            raise NameError('No File at %s'%path)

    def get_pts(self):
        return self.pts

class BBox2DExtractor:
    def __init__(self, pov_path, camera_matrix, bbox3D):
        self.k = camera_matrix
        self.bbox = bbox3D
        if path.exists(pov_path):
            data = None
            with open(pov_path) as f:
                    data = json.load(f)
            if data != None:
                T = data['sensor']['T_Mat']
                self.T = np.array(T)
                self.T = utils.changeHand(self.T)
            else:
                raise NameError('No data in %s'%path)
        else:
            raise NameError('No File at %s'%path)

    def project_3Dbbox(self):
        T_CCw = utils.getTCCw()
        T_CwW = np.linalg.inv(self.T)
        T_CW = np.matmul(T_CCw, T_CwW)
        vertex = []
        for pt in self.bbox.get_pts():
            point = pt
            in_cam_space = np.matmul(T_CW, point) # place the point in camera space
            in_cam_space_cropped = in_cam_space[:3] # crop to get a 1x3 vector
            in_px_raw = np.matmul(self.k, in_cam_space_cropped) # project the point in the camera plane
            in_px_norm = in_px_raw / in_px_raw[2] # normalize the projection
            # if in_px_raw[2] < 0.0: # if the point is behind the camera (z < 0), return an empty set of points
                # return []
            vertex.append(in_px_norm) 
        return vertex

    def get_2Dbbox(self):
        vertex = self.project_3Dbbox()
        v = np.transpose(vertex)
        x_max = int(np.amax(v[0]))
        x_min = int(np.amin(v[0]))
        y_max = int(np.amax(v[1]))
        y_min = int(np.amin(v[1]))
        out = []
        out.append(np.array([x_min, y_min]))
        out.append(np.array([x_max, y_min]))
        out.append(np.array([x_max, y_max]))
        out.append(np.array([x_min, y_max]))
        return out

if __name__ == '__main__':
    FRAME = 154
    PATH = "/home/caillot/Documents/Dataset/CARLA_Dataset_A/Embed/V%d/VehicleInfo/%06d.json"
    p2 = "/home/caillot/Documents/Dataset/CARLA_Dataset_A/Infra/sensorInfo/%06d.json"%(FRAME)
    path_to_npy = "/home/caillot/Documents/Dataset/CARLA_Dataset_A/Infra/cameraRGB/cameraMatrix.npy"
    img = cv.imread("/home/caillot/Documents/Dataset/CARLA_Dataset_A/Infra/cameraRGB/%06d.png"%(FRAME))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cameraMatrix = np.load(path_to_npy)
    color = (0, 255, 0)
    thickness = 2

    for i in range(3):
        bbox = BBox3DExtractor(PATH%(i, FRAME))
        print(bbox.get_pts())
        bbox2d = BBox2DExtractor(p2, cameraMatrix, bbox)
        v = bbox2d.get_2Dbbox()
        print(v)

        for i in range(len(v)):
            img = cv.line(img, v[i], v[(i+1)%len(v)], color, thickness)

    plt.imshow(img)
    plt.show()