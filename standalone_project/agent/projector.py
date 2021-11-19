import numpy as np
from numpy.core.fromnumeric import size
from numpy.core.numeric import moveaxis
from vector import *
from bbox import *
from Tmat import TMat
import cv2 as cv
from copy import deepcopy
import matplotlib.pyplot as plt

import open3d as o3d

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

def projector_filter(bbox:Bbox3D, vPose:TMat, k:TMat, cwTw:TMat, img_path:str, threashold:float = 0.3) -> Bbox2D:
    out_bbox = Bbox2D(vec2(0, 0), vec2(5, 5), label=bbox.get_label())
    img = cv.imread(img_path)
    (h, w, c) = img.shape
    
    cwTc = getCwTc()
    wTc = cwTw * cwTc
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

    # vehicle : 10
    # pedestrian : 4
    try:
        todetect = 10 if bbox.get_label() == "vehicle" else 4
        unique, counts = np.unique(cropped_img, return_counts=True)
        pix = dict(zip(unique, counts))
        N_detected = pix[todetect]
        ratio = N_detected / cropped_img.size
        print(f'In {cropped_img.size} pix, deteceted {pix} with {N_detected} of {bbox.get_label()} with a ratio of {ratio*100.0}%')
    except:
        return None
    if ratio <= threashold:
        return None



    # cv.imshow('image', )
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # plt.imshow(cropped_img)
    # plt.show()
    
    return out_bbox


if __name__ == '__main__':
    # img_path = '/home/caillot/Documents/Dataset/CARLA_Dataset_B/I000/camera_semantic_segmentation/001769.png'
    img_path = '/home/caillot/Documents/Dataset/CARLA_Dataset_A/Infra/cameraRGB/000051.png'
    # <4.553, 2.097, 1.767> @ <-2.308, -1.049, 0.007>
    # <2.207, 1.481, 1.376> @ <2.242, -43.123, -0.017>
    # bbox3d = Bbox3D(pose=vec3(2.242, -43.123, -0.017), size=vec3(2.207, 1.481, 1.376), label="vehicle")
    bbox3d = Bbox3D(pose=vec3(0.0, 0.0, 0.76), size=vec3(4.553, 2.097, 1.767), label="vehicle")
    # k = load_k('/home/caillot/Documents/Dataset/CARLA_Dataset_B/I000/camera_semantic_segmentation/cameraMatrix.npy')
    k = load_k('/home/caillot/Documents/Dataset/CARLA_Dataset_A/Infra/cameraRGB/cameraMatrix.npy')
    print(k)

    camtmat = [[-4.22219498e-08,  1.00000000e+00, -2.03222601e-06,  0.00000000e+00], [-9.65925851e-01, -5.66761877e-07, -2.58818951e-01, -0.00000000e+00], [-2.58818951e-01,  1.95205180e-06,  9.65925851e-01,  1.30000000e+01], [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
    camtmat = [[-4.10752676316406e-08, -1.0, -1.4950174431760388e-08, 0.0], [0.9396926164627075, -4.371138828673793e-08, 0.3420201241970062, 0.0], [-0.3420201241970062, -0.0, 0.9396926164627075, 13.0], [0.0, 0.0, 0.0, 1.0]]
    camtmat = np.array(camtmat)
    wTcw = TMat()
    wTcw.set(camtmat)

    # map size
    map_size = 70

    a = np.array([[0.01935034990310669, 0.9998127818107605, 0.0, 4.926102638244629], [-0.9998127818107605, 0.01935034990310669, 0.0, 40.57860565185547], [0.0, -0.0, 1.0, -0.024867916479706764], [0.0, 0.0, 0.0, 1.0]])
    a = np.array([[-0.03497249260544777, -0.9993882775306702, 0.0, -6.446169853210449], [0.9993882775306702, -0.03497249260544777, -0.0, -42.193748474121094], [0.0, -0.0, 1.0, 0.035129792988300323], [0.0, 0.0, 0.0, 1.0]])
    a= np.array([[0.01935034990310669, 0.9998127818107605, 0.0, 4.926102638244629], [-0.9998127818107605, 0.01935034990310669, 0.0, 40.57860565185547], [0.0, -0.0, 1.0, 0.03291169926524162], [0.0, 0.0, 0.0, 1.0]])
    vMat = TMat()
    vMat.set(a)
    

    wTcw.handinessLeft2Right()
    vMat.handinessLeft2Right()

    bbox = projector_filter(bbox3d, vMat, k, wTcw, img_path)


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

    # cwTc = getCwTc()
    # wTc = wTcw * cwTc
    # # wTc = wTcw



    # # World ref
    # mesh_world_center = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])

    # # Camera ref
    # mesh_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # mesh_camera.transform(wTc.get())


    # mesh_camera2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # mesh_camera2.transform(wTc.get())

    # # Car point (at center)
    # mesh_car = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    # mesh_car.paint_uniform_color([.1, .8, .1])
    # mesh_car.transform(vMat.get())

    # # Drawing the ground with a grid
    # # Red for X
    # # Green for Y
    # x_col = [0.5, 0.5, 0.5]
    # y_col = [0.5, 0.5, 0.5]
    # pointsX = []
    # pointsY = []
    # lineX = []
    # lineY = []

    # for i in range(-map_size, map_size, 1):
    #     pointsX.append([i, -map_size, 0])
    #     pointsX.append([i, map_size, 0])


    # for i in range(0, len(pointsX), 2):
    #     lineX.append([i, i+1])

    # for i in range(-map_size, map_size, 1):
    #     pointsY.append([-map_size, i, 0])
    #     pointsY.append([map_size, i, 0])

    # for i in range(0, len(pointsX), 2):
    #     lineY.append([i, i+1])

    # colorsX = [x_col for i in range(len(lineX))]
    # line_setX = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(pointsX),
    #     lines=o3d.utility.Vector2iVector(lineX)
    # )
    # line_setX.colors = o3d.utility.Vector3dVector(colorsX)

    # colorsY = [y_col for i in range(len(lineY))]
    # line_setY = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(pointsY),
    #     lines=o3d.utility.Vector2iVector(lineY)
    # )
    # line_setY.colors = o3d.utility.Vector3dVector(colorsY)

    # o3d.visualization.draw([mesh_world_center, line_setX, line_setY, mesh_camera2, mesh_car])