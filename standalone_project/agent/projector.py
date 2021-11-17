import numpy as np
from numpy.core.fromnumeric import size
from vector import *
from bbox import *
from Tmat import TMat
import cv2 as cv
from copy import deepcopy

def load_k(path_k) -> TMat:
    k = np.load(path_k)
    kmat = TMat()
    kmat4 = np.identity(4)
    kmat4[:3, :3] = k
    # kmat4[:2, :2] = k[:2, :2]
    # kmat4[:2, 3] = k[:2, 2]
    kmat.set(kmat4)
    return kmat

def projector_filter(bbox:Bbox3D, k:TMat, camPose:TMat, img_path:str) -> Bbox2D:
    img = cv.imread(img_path)
    (h, w, c) = img.shape
    del img
    poseCam = deepcopy(camPose)
    print(poseCam)
    poseCam.inv()
    print(poseCam)
    ptsin:List[vec4] = [pt3.vec4() for pt3 in bbox.get_pts()]
    print(ptsin[0])
    ptsin:List[vec4] = [poseCam * pt4 for pt4 in ptsin]
    print(ptsin[0])
    ptsout:List[vec4] = [k * pt4 for pt4 in ptsin]
    print(ptsout[0])
    ptsout:List[vec3] = [pt4.vec3() for pt4 in ptsout]
    print(ptsout[0])
    ptsout:List[vec2] = [pt4.nvec2() for pt4 in ptsout]
    print(ptsout[0])

    out_bbox = Bbox2D(None, None, label=bbox.get_label())
    out_bbox.set_from_pts(ptsout)
    print(out_bbox)

    center:vec2 = out_bbox.get_pose() + (out_bbox.get_size() / 2.0)

    print(center.__str__() + f' in <{w}; {h}>')
    if center.x() >= 0.0 and center.x() <= w and center.y() >= 0.0 and center.y() <= h:
        return out_bbox
    else:
        return None
    


    # cv.imshow('image',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


if __name__ == '__main__':
    img_path = '/home/caillot/Documents/Dataset/CARLA_Dataset_B/I000/camera_semantic_segmentation/001769.png'
    # <4.553, 2.097, 1.767> @ <-2.308, -1.049, 0.007>
    # <2.207, 1.481, 1.376> @ <2.242, -43.123, -0.017>
    # bbox3d = Bbox3D(pose=vec3(2.242, -43.123, -0.017), size=vec3(2.207, 1.481, 1.376), label="vehicle")
    bbox3d = Bbox3D(pose=vec3(-2.308, -1.049, 0.007), size=vec3(4.553, 2.097, 1.767), label="vehicle")
    k = load_k('/home/caillot/Documents/Dataset/CARLA_Dataset_B/I000/camera_semantic_segmentation/cameraMatrix.npy')
    print(k)

    camtmat = [[-4.22219498e-08,  1.00000000e+00, -2.03222601e-06,  0.00000000e+00], [-9.65925851e-01, -5.66761877e-07, -2.58818951e-01, -0.00000000e+00], [-2.58818951e-01,  1.95205180e-06,  9.65925851e-01,  1.30000000e+01], [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
    camtmat = np.array(camtmat)
    cTMat = TMat()
    cTMat.set(camtmat)

    print(projector_filter(bbox3d, k, cTMat, img_path))