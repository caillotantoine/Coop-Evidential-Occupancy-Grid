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
import numpy as np
import cv2

from nav_msgs.msg import OccupancyGrid

class Projector:

    # Attributes
    #
    #   cam_pos_mat : 4x4 transform matrix
    #   K : camera matrix
    #   gnd_plane : Plucker's coordinates systeme based ground plane
    #   worked_ROI : table of tuple (pts_bbox, footprint_pts)
    #       pts_bbox : 
    #       footprint_pts : 

    def __init__(self, bboxes: BBox2D, gnd_plane = (vec4n(0, 0, 0), vec4n(1, 1, 0), vec4n(1, 0, 0)), step_grid = 0.1, grid_range = 75):
        self.step_grid = step_grid
        self.grid_range = grid_range
        self.grid_size = int(2*self.grid_range/self.step_grid)
        self.cam_pos_mat = pose2Tmat(bboxes.cam_pose) # get the camera position in the world
        self.cam_pos_mat = np.matmul(self.cam_pos_mat, np.linalg.inv(getTCCw())) # apply transformation of axis to get a transformation image -> world
        self.K = np.reshape(np.array(bboxes.cam_info.K), (3, 3)) # get the camera matrix
        # rospy.logerr("K: \n{}".format(self.K))
        self.K_inv = np.linalg.inv(self.K)
        ROIs :RegionOfInterest = bboxes.roi # get every ROI given from that camera
        (A, B, C) = gnd_plane # extract the 3 vectors defining the ground plane
        self.gnd_plane = plucker.plane(A, B, C) # define a ground plane in plucker coordinates
        self.cam = vec4n(0, 0, 0)
        self.cam = np.matmul(self.cam_pos_mat, self.cam)

        self.worked_ROI = []

        for roi in ROIs: # For each ROI
            vertex = [] # extract each 
            footprint_pts = []
            pts_bbox = []
            pts_bbox.append([self.cam[i][0] for i in range(3)])

            vertex.append(vec3(roi.x_offset, roi.y_offset, 1))
            vertex.append(vec3(roi.x_offset + roi.width, roi.y_offset, 1))
            vertex.append(vec3(roi.x_offset + roi.width, roi.y_offset + roi.height, 1))
            vertex.append(vec3(roi.x_offset, roi.y_offset + roi.height, 1))
            for p in vertex:
                ps = np.matmul(self.K_inv, p) * 60.0
                # rospy.logwarn("PS : \n{}".format(ps))
                pg = np.matmul(self.cam_pos_mat, vec3tovec4(ps))
                # pts_bbox.append([pg[i][0] for i in range(3)])
                L = plucker.line(self.cam, pg)
                fp_pt = plucker.interLinePlane(L, self.gnd_plane)
                pts_bbox.append([normVec4(fp_pt)[i][0] for i in range(3)])
                footprint_pts.append(normVec4(fp_pt))

            self.worked_ROI.append((pts_bbox, footprint_pts))


    def get_geometries(self):
        meshes = []
        # World ref
        # mesh_world_center = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

        # Camera ref
        mesh_camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh_camera.transform(self.cam_pos_mat)
        meshes.append(mesh_camera)

        for roi in self.worked_ROI:
            (pts_bbox, footprint_pts) = roi
            # Draw the lines from the camera center corresponding to the BBox
            lines_bbox = []
            for i in range(1, len(pts_bbox)):
                lines_bbox.append([0, i])

            colors_lines_bbox = [[0, 0, 1] for i in range(len(lines_bbox))]
            lineset_bbox = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts_bbox),
                lines=o3d.utility.Vector2iVector(lines_bbox)
            )
            lineset_bbox.colors = o3d.utility.Vector3dVector(colors_lines_bbox)

            meshes.append(lineset_bbox)

            # Draw the footprint
            fp_points_td = []
            for p in footprint_pts:
                fp_points_td.append([p[i][0] for i in range(3)])

            fp_lines_td = []
            for i in range(len(fp_points_td)):
                fp_lines_td.append([i, (i+1)%len(fp_points_td)])

            colors_lines_fp = [[1, 0, 0] for i in range(len(fp_lines_td))]
            lineset_bbox_fp = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(fp_points_td),
                lines=o3d.utility.Vector2iVector(fp_lines_td)
            )
            lineset_bbox_fp.colors = o3d.utility.Vector3dVector(colors_lines_fp)

            meshes.append(lineset_bbox_fp)
        return meshes
    
    def get_occupGrid(self):
        rawGOL = OccupancyGrid()
        rawGOL.info.resolution = self.step_grid
        rawGOL.info.width = self.grid_size
        rawGOL.info.height = self.grid_size

        GOL_origin = Pose()
        GOL_origin.position.x = - (self.grid_size * self.step_grid / 2.0)
        GOL_origin.position.y = - (self.grid_size * self.step_grid / 2.0)
        GOL_origin.position.z = 0

        rawGOL.info.origin = GOL_origin

        raw_map = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        # raw_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        # rospy.logwarn("map size : {}".format(self.grid_size))

        for roi in self.worked_ROI:
            (pts_bbox, footprint_pts) = roi
            pts = []
            for pt in footprint_pts:
                x = int(pt[0][0] / self.step_grid + (self.grid_size / 2.0))
                y = int(pt[1][0] / self.step_grid + (self.grid_size / 2.0))
                pts.append([x, y])
            rospy.logwarn("footprint \n{}".format(np.array(pts)))
            cv2.fillPoly(raw_map, pts=[np.array(pts)], color=100)




        rawGOL.data = raw_map.flatten().tolist()

        return rawGOL


