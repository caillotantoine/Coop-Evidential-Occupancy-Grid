
import BoundingBoxExtractor
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

path_dataset = "/home/caillot/Documents/Dataset/CARLA_Dataset_A"
path_vehicle = "/Embed/V%d"
path_infra = "/Infra"
path_v_json = "/VehicleInfo/%06d.json"
path_i_json = "/sensorInfo/%06d.json"
path_rgb = "/cameraRGB/%06d.png"
path_K = "/cameraRGB/cameraMatrix.npy"

start_frame = 61
end_frame = 577
start_v = 0
N_v = 3

frame = 152
v = 1 



img = cv.imread(path_dataset+path_infra+(path_rgb%(frame)))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cameraMatrix = np.load(path_dataset+path_infra+path_K)
color = (0, 255, 0)
thickness = 2
for i in range(3):
    p_bbox = path_dataset+(path_vehicle%(i))+(path_v_json%(frame))
    p_pov = path_dataset+path_infra+(path_i_json%(frame))   
    bbox = BoundingBoxExtractor.BBox3DExtractor(p_bbox)
    print(bbox.get_pts())
    bbox2d = BoundingBoxExtractor.BBox2DExtractor(p_pov, cameraMatrix, bbox)
    v = bbox2d.get_2Dbbox()
    print(v)

    for i in range(len(v)):
        img = cv.line(img, v[i], v[(i+1)%len(v)], color, thickness)

plt.imshow(img)
plt.show()