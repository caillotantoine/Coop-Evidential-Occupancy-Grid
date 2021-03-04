from pyquaternion import Quaternion
import numpy as np


def pose2Tmat(pose):
    # get the position of the camera
    R_Q = Quaternion(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z) 
    T_mat = R_Q.transformation_matrix   # Transfromation only fitted with rotation elements
    T_mat[0][3] = pose.position.x   # Adding the translation elements
    T_mat[1][3] = pose.position.y
    T_mat[2][3] = pose.position.z
    return T_mat

def getTCCw():
    return np.array([[0.0,   1.0,   0.0,   0.0,], [0.0,   0.0,  -1.0,   0.0,], [1.0,   0.0,   0.0,   0.0,], [0.0,   0.0,   0.0,   1.0,]])
    # matrix to change from world space to camera space 