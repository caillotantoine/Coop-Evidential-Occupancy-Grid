import rospy
import numpy as np
import cv2 as cv
from threading import Lock
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pyquaternion import Quaternion
from perceptive_stream.msg import Img, BBox3D
from geometry_msgs.msg import Pose


def pose2Tmat(pose):
    # get the position of the camera
    R_Q = Quaternion(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z) 
    T_mat = R_Q.transformation_matrix   # Transfromation only fitted with rotation elements
    T_mat[0][3] = pose.position.x   # Adding the translation elements
    T_mat[1][3] = pose.position.y
    T_mat[2][3] = pose.position.z
    return T_mat


def Tmat2pose (TMat):
    out = Pose()
    outT = np.array(TMat)
    out_t = outT[:3, 3:4]
    out.position.x = out_t.flatten()[0]
    out.position.y = out_t.flatten()[1]
    out.position.z = out_t.flatten()[2]

    out_r = outT[:3, :3]

    # T_Mat only contains pure rotation and traslation. Therefore we filter the float approx.
    Ry = -np.arcsin(out_r[2, 0])
    Ry = np.pi - Ry if abs(np.cos(Ry) - 0.0) < 0.01 else Ry
    Rz = np.arctan2(out_r[1,0]/np.cos(Ry), out_r[0,0]/np.cos(Ry))
    Rx = np.arctan2(out_r[2,1]/np.cos(Ry), out_r[2,2]/np.cos(Ry))

    # rospy.loginfo("R[x, y, z] : %f°, %f°, %f°" % (np.degrees(Rx),np.degrees(Ry),np.degrees(Rz)))

    Qx = Quaternion(axis=[1.0, 0.0, 0.0], radians=Rx)
    Qy = Quaternion(axis=[0.0, 1.0, 0.0], radians=Ry)
    Qz = Quaternion(axis=[0.0, 0.0, 1.0], radians=Rz)
    out_R = Qx * Qy * Qz

    out.orientation.w = out_R.w
    out.orientation.x = out_R.x
    out.orientation.y = out_R.y
    out.orientation.z = out_R.z

    return out

def getTCCw():
    return np.array([[0.0,   1.0,   0.0,   0.0,], [0.0,   0.0,  -1.0,   0.0,], [1.0,   0.0,   0.0,   0.0,], [0.0,   0.0,   0.0,   1.0,]])
    # matrix to change from world space to camera space 