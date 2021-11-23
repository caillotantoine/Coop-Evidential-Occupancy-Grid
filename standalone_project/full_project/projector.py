import numpy as np
from numpy.core.fromnumeric import size
from numpy.core.numeric import moveaxis
from scipy.spatial import transform
from vector import *
from bbox import *
from Tmat import TMat
import cv2 as cv
from copy import deepcopy
import matplotlib.pyplot as plt
from plucker import plkrLine, plkrPlane

import open3d as o3d

def get_o3dgrid(map_size = 70):
    # Drawing the ground with a grid
    # Red for X
    # Green for Y
    x_col = [0.5, 0.5, 0.5]
    y_col = [0.5, 0.5, 0.5]
    pointsX = []
    pointsY = []
    lineX = []
    lineY = []

    for i in range(-map_size, map_size, 1):
        pointsX.append([i, -map_size, 0])
        pointsX.append([i, map_size, 0])


    for i in range(0, len(pointsX), 2):
        lineX.append([i, i+1])

    for i in range(-map_size, map_size, 1):
        pointsY.append([-map_size, i, 0])
        pointsY.append([map_size, i, 0])

    for i in range(0, len(pointsX), 2):
        lineY.append([i, i+1])

    colorsX = [x_col for i in range(len(lineX))]
    line_setX = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pointsX),
        lines=o3d.utility.Vector2iVector(lineX)
    )
    line_setX.colors = o3d.utility.Vector3dVector(colorsX)

    colorsY = [y_col for i in range(len(lineY))]
    line_setY = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pointsY),
        lines=o3d.utility.Vector2iVector(lineY)
    )
    line_setY.colors = o3d.utility.Vector3dVector(colorsY)
    return [line_setX, line_setY]

def getCwTc():
    out = TMat()
    # matout = np.array([[0.0,   -1.0,   0.0,   0.0,], [0.0,   0.0,  -1.0,   0.0,], [1.0,   0.0,   0.0,   0.0,], [0.0,   0.0,   0.0,   1.0,]])
    matout = np.array([[0.0,   0.0,   1.0,   0.0,], [-1.0,   0.0,  0.0,   0.0,], [0.0,   -1.0,   0.0,   0.0,], [0.0,   0.0,   0.0,   1.0,]])
    out.set(matout)
    return out

def load_k(path_k) -> TMat:
    k = np.load(path_k)
    kmat = TMat()
    kmat4 = np.identity(4)
    kmat4[:3, :3] = k
    kmat.set(kmat4)
    return kmat

def projector_filter(bbox:Bbox3D, vPose:TMat, k:TMat, sensorT:TMat, img, threashold:float = 0.3) -> Bbox2D:
    out_bbox = Bbox2D(vec2(0, 0), vec2(5, 5), label=bbox.get_label())
    
    
    cwTc = getCwTc()
    wTc = sensorT * cwTc
    wTc.inv()

    pts_c:List[vec4] = [pt3.vec4() for pt3 in bbox.get_pts()]
    pts_w:List[vec4] = [vPose * pt4 for pt4 in pts_c]
    pts_cam:List[vec4] = [wTc * pt4 for pt4 in pts_w]
    pts_proj:List[vec4] = [k * pt4 for pt4 in pts_cam]
    pts_2d:List[vec2] = [pt3.nvec2() for pt3 in [pt4.vec3() for pt4 in pts_proj]]

    for i in range(len(pts_2d)):
        if pts_proj[i].z() <= 0:
            return None

    out_bbox.set_from_pts(pts_2d)

    (h, w, c) = img.shape
    center:vec2 = out_bbox.get_pose() + (out_bbox.get_size() / 2.0)
    if not (center.x() >= 0.0 and center.x() <= w and center.y() >= 0.0 and center.y() <= h):
        return None
    #     pass
    #     # return out_bbox
    # else:
    #     return None

    posebbox = out_bbox.get_pose()
    sizebbox = out_bbox.get_size()

    cropped_img = img[int(posebbox.y()):int(posebbox.y()+sizebbox.y()), int(posebbox.x()):int(posebbox.x()+sizebbox.x()), 2]
    
    try:
        # vehicle : 10
        # pedestrian : 4
        todetect = 10 if bbox.get_label() == "vehicle" else 4
        unique, counts = np.unique(cropped_img, return_counts=True)
        pix = dict(zip(unique, counts))
        N_detected = pix[todetect]
        ratio = N_detected / cropped_img.size
        # print(f'In {cropped_img.size} pix, deteceted {pix} with {N_detected} of {bbox.get_label()} with a ratio of {ratio*100.0}%')
    except:
        return None
    if ratio <= threashold:
        return None
    return out_bbox

def project_BBox2DOnPlane(plane:plkrPlane, bbox:Bbox2D, kMat:TMat, sensorT:TMat, vMat:TMat = None, vbbox3d:Bbox3D = None, debug = None) -> List[vec2]:
    invK = deepcopy(kMat)
    invK.inv()
    print(invK)

    cwTc = getCwTc()
    wTcw = sensorT
    wTc = wTcw * cwTc
    

    mesh_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    mesh_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    mesh_camera.transform(wTc.get())

    pts4 = [pt2.vec4(z=1) for pt2 in bbox.get_pts()]
    pts_ip_c = [(invK * p) for p in pts4]
    pts_ip_cw = [(wTc * pt4) for pt4 in pts_ip_c]
    lines = [plkrLine(wTc.get_translation(), pt4) for pt4 in pts_ip_cw]
    out_pts = [plane.intersect(line) for line in lines]
    for pt in out_pts:
        pt.normalize()
        if debug != None:
            print(pt)
    
    out_pts = [pt4.vec3() for pt4 in out_pts]
    out_pts = [pt3.vec2() for pt3 in out_pts]
    if debug == None:
        return out_pts

    mesh_ip_c = [o3d.geometry.TriangleMesh.create_sphere(radius=0.1) for pt in pts_ip_c]
    for idx, mesh in enumerate(mesh_ip_c):
        mesh.paint_uniform_color([0.1, 0.1, 0.8])
        T = np.identity(4)
        T[:3, 3] = np.transpose(pts_ip_c[idx].get())[0,:3]
        # print(T)
        mesh.transform(T)

    
    mesh_ip_cw = [o3d.geometry.TriangleMesh.create_sphere(radius=0.1) for pt in pts_ip_cw]
    for idx, mesh in enumerate(mesh_ip_cw):
        mesh.paint_uniform_color([0.8, 0.5, 0.1])
        T = np.identity(4)
        T[:3, 3] = np.transpose(pts_ip_cw[idx].get())[0,:3]
        # print(T)
        mesh.transform(T)

    mesh_out_pts = [o3d.geometry.TriangleMesh.create_sphere(radius=0.5) for pt in out_pts]
    for idx, mesh in enumerate(mesh_out_pts):
        mesh.paint_uniform_color([0.8, 0.1, 0.1])
        T = np.identity(4)
        T[:3, 3] = np.transpose(out_pts[idx].get())[0,:3]
        # print(T)
        mesh.transform(T)

    carMesh = None
    if(vMat != None):
        if(vbbox3d == None):
            carMesh = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        else:
            carMesh = o3d.geometry.TriangleMesh.create_box(width=vbbox3d.get_size().x(), height=vbbox3d.get_size().z(), depth=vbbox3d.get_size().y())
            T = np.identity(4)
            T[0, 3] = -vbbox3d.get_size().x()/2.0 + vbbox3d.get_pose().x()
            T[1, 3] = -vbbox3d.get_size().y()/2.0 + vbbox3d.get_pose().y()
            T[2, 3] = -vbbox3d.get_size().z()/2.0 + vbbox3d.get_pose().z()
            carMesh.transform(T)
    carMesh.transform(vMat.get())
    carMesh.paint_uniform_color([0.0, 0.5, 0.1])

    o3d.visualization.draw(get_o3dgrid() + [mesh_world, mesh_camera] + mesh_ip_c + mesh_ip_cw + [carMesh] + mesh_out_pts)
    return out_pts


if __name__ == '__main__':
    img_path = '/home/caillot/Documents/Dataset/CARLA_Dataset_B/I000/camera_rgb/000072.png'
    bbox3d = Bbox3D(pose=vec3(0.0, 0.0, 0.76), size=vec3(4.553, 2.097, 1.767), label="vehicle")
    k = load_k('/home/caillot/Documents/Dataset/CARLA_Dataset_B/I000/camera_rgb/cameraMatrix.npy')
    print(k)

    camtmat = [[-4.222195926217864e-08, -1.0, -2.032226348092081e-06, 0.0], [0.9659258723258972, -5.667619689120329e-07, 0.25881895422935486, 0.0], [-0.25881895422935486, -1.952052116394043e-06, 0.9659258723258972, 13.0], [0.0, 0.0, 0.0, 1.0]]
    camtmat = np.array(camtmat)
    wTcw = TMat()
    wTcw.set(camtmat)

    a= np.array([[0.06367591768503189, 0.9979698657989502, 0.0012183079961687326, 5.033266544342041], [-0.9979450106620789, 0.06368298083543777, -0.007091572042554617, 39.51407241821289], [-0.007154760882258415, -0.0007642419659532607, 0.9999741315841675, 0.006778659764677286], [0.0, 0.0, 0.0, 1.0]])
    vMat = TMat()
    vMat.set(a)
    
    wTcw.handinessLeft2Right()
    vMat.handinessLeft2Right()
        
    img2 = cv.imread(f'/home/caillot/Documents/Dataset/CARLA_Dataset_B/I000/camera_semantic_segmentation/000072.png')
    # bbox = projector_filter(bbox3d, vMat, k, wTcw, img)
    bbox = projector_filter(bbox3d, vMat, k, wTcw, img2, 0.2)

    img = cv.imread(img_path)
    color = (0, 255, 0)
    thickness = 2

    points = bbox.get_pts()
    pts = [tuple(np.transpose(pt.get())[0].astype(int).tolist()) for pt in points]
    print(pts)
    for i in range(len(pts)):
        img = cv.line(img, pts[i], pts[(i+1)%len(pts)], color, thickness)

    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    ground = plkrPlane()
    
    traces = project_BBox2DOnPlane(ground, bbox, k, wTcw, vMat=vMat, vbbox3d=bbox3d)
    for t in traces:
        print(t)
    
    