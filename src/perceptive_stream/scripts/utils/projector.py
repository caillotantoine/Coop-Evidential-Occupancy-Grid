#!/usr/bin/env python3
# coding: utf-8

import rospy
import open3d as o3d
from sensor_msgs.msg import RegionOfInterest, CameraInfo
from geometry_msgs.msg import Pose
# from scipy.spatial.transform import Rotation as R
from perceptive_stream.msg import BBox2D
from utils.utilsant import pose2Tmat, getTCCw
from utils.vecMatUtils import *
from utils import plucker

class Projector:

    # Attributes
    #
    #   cam_pos_mat : 4x4 transform matrix
    #   K : camera matrix
    #   gnd_plane : Plucker's coordinates systeme based ground plane
    #   

    def __init__(self, bboxes: BBox2D, gnd_plane = (vec4n(0, 0, 0), vec4n(1, 1, 0), vec4n(1, 0, 0))):
        self.cam_pos_mat = pose2Tmat(bboxes.cam_pose) # get the camera position in the world
        self.cam_pos_mat = np.matmul(self.cam_pos_mat, np.linalg.inv(getTCCw()))
        self.K = bboxes.cam_info.K # get the camera matrix
        ROIs :RegionOfInterest = bboxes.roi # get every ROI given from that camera
        (A, B, C) = gnd_plane # extract the 3 vectors defining the ground plane
        self.gnd_plane = plucker.plane(A, B, C) # define a ground plane in plucker coordinates

        for roi in ROIs: # For each ROI
            vertex = [] # extract each 
            vertex.append(vec3(roi.x_offset, roi.y_offset, 1))
            vertex.append(vec3(roi.x_offset + roi.width, roi.y_offset, 1))
            vertex.append(vec3(roi.x_offset + roi.width, roi.y_offset + roi.height, 1))
            vertex.append(vec3(roi.x_offset, roi.y_offset + roi.height, 1))


    def get_geometries(self):
        # World ref
        mesh_world_center = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

        # Camera ref
        mesh_camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh_camera.transform(self.cam_pos_mat)

        return [mesh_camera]

