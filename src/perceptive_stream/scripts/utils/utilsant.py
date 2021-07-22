#!/usr/bin/env python3
# coding: utf-8

import rospy
import numpy as np
from sensor_msgs.msg import Image
from pyquaternion import Quaternion
from perceptive_stream.msg import Img, BBox3D
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R


def pose2Tmat(pose):
    # get the position of the camera
    R_Q = Quaternion(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z) 
    T_mat = R_Q.transformation_matrix   # Transfromation only fitted with rotation elements
    T_mat[0][3] = pose.position.x   # Adding the translation elements
    T_mat[1][3] = pose.position.y
    T_mat[2][3] = pose.position.z
    return T_mat


def Tmat2pose (TMat, changeHandLeft2Right = False):
    out = Pose()
    outT = np.array(TMat)
    out_t = outT[:3, 3:4]
    out.position.x = out_t.flatten()[0]
    out.position.y = out_t.flatten()[1]
    out.position.z = out_t.flatten()[2]

    r = R.from_dcm(outT[:3, :3])
    r_euler = R.as_euler(r, 'xyz')

    if changeHandLeft2Right:
        # Code de Yohan pour le changement de main
        out.position.y = -out.position.y
        r_euler[0]=- r_euler[0]
        r_euler[2]=- r_euler[2]

    r_out = R.from_euler('xyz', r_euler).as_quat() # Format: X Y Z W
    out.orientation.w = r_out[3]
    out.orientation.x = r_out[0]
    out.orientation.y = r_out[1]
    out.orientation.z = r_out[2]

    return out

def getTCCw():
    return np.array([[0.0,   -1.0,   0.0,   0.0,], [0.0,   0.0,  -1.0,   0.0,], [1.0,   0.0,   0.0,   0.0,], [0.0,   0.0,   0.0,   1.0,]])
    # matrix to change from world space to camera space 

