#!/usr/bin/env python3
# coding: utf-8


# /!\ THIS FILE IS NOT A ROS NODE



import numpy as np
import cv2
import json
import argparse
from os import path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm

NV = 3
FRAME = 154
PATH = "/home/caillot/Documents/Dataset/CARLA_Dataset_A/Embed/V%d/VehicleInfo/%06d.json"
PATH_OUT = ""


def changeHand(mat):
    T = mat[:3, 3:4]
    r = R.from_dcm(mat[:3, :3])
    r_euler = R.as_euler(r, 'xyz')
    T[1] = -T[1]
    r_euler[0]=- r_euler[0]
    r_euler[2]=- r_euler[2]
    r_out = R.from_euler('xyz', r_euler).as_dcm()
    out = np.identity(4)
    out[:3, 3:4] = T
    out[:3, :3] = r_out
    return out

def getBBox_points(p):
    if path.exists(p):
        data = None
        with open(p) as f:
                data = json.load(f)
        if data != None:
            l = data['vehicle']['BoundingBox']['extent']['x']
            w = data['vehicle']['BoundingBox']['extent']['y']
            s = np.array([l, w])
            vT = changeHand(np.array(data['vehicle']['T_Mat']))
            pts = []

            for i in [0,1,3,2]:
                mask = np.array(list(map(int,list("{0:02b}".format(i)))))
                mask = np.where(mask==1,1,-1)
                vec = s * mask
                vecout = np.zeros(4)
                vecout[:2] = vec
                vecout[3] = 1

                pt = np.matmul(vT, vecout)
                pt = [pt[0], pt[1]]
                pts.append(pt)
            return pts
        else:
            raise NameError('No bounding box info in %s'%path)
    else:
        raise NameError('No File at %s'%path)

def draw(pts_lst, grid_size, step_grid):
    raw_map = np.full((grid_size, grid_size), 0, dtype=np.float)
    data = {}
    data['roi'] = []
    for i, pts_bbox in enumerate(pts_lst):
        pts=[]
        for pt in pts_bbox:
            x = int(pt[0] / step_grid + (grid_size / 2.0))
            y = int(pt[1] / step_grid + (grid_size / 2.0)) 
            pts.append([x, y])
        pts_t = np.transpose(np.array(pts))
        radius = 1.5
        x_max = np.amax(pts_t[0]) + (radius / step_grid)
        x_min = np.amin(pts_t[0]) - (radius / step_grid)
        y_max = np.amax(pts_t[1]) + (radius / step_grid)
        y_min = np.amin(pts_t[1]) - (radius / step_grid)
        data['roi'].append({
            'vehicle': i,
            'xmin': x_min,
            'xmax': x_max,
            "ymin": y_min,
            "ymax": y_max
        })
        cv2.fillPoly(raw_map, pts=[np.array(pts)], color=100)
    return (raw_map, data)


def main(path2json, nv, frame, out="", step_grid=0.2, grid_range=75, outjsonpath=""):
    grid_size = int(2*grid_range/step_grid)
    pts_lst = []
    for v in range(nv):
        jsonPath = path2json%(v, frame)
        pts_lst.append(getBBox_points(jsonPath))
    (gog, jsonout) = draw(pts_lst, grid_size, step_grid)
    plt.imsave(out, gog, vmin=-1, vmax=100, cmap='Greys')
    with open(outjsonpath, 'w') as outfile:
        json.dump(jsonout, outfile)


if __name__ == '__main__':
    p = "/home/caillot/Bureau/gog_gnd_truth/%06d.png"%FRAME
    print("Starting generation")
    for i in tqdm(range(61, 577)):
        p = "/home/caillot/Bureau/gog_gnd_truth/img/%06d.png"%i
        p2 = "/home/caillot/Bureau/gog_gnd_truth/json/%06d.json"%i
        main(PATH, NV, i, out=p, outjsonpath=p2)
    print("Generation finished")