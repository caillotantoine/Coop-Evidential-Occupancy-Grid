#!/usr/bin/env python3
# coding: utf-8

from codecs import xmlcharrefreplace_errors
from typing import NamedTuple
from numpy.core.numerictypes import obj2sctype
import rospy
import open3d as o3d
from sensor_msgs.msg import RegionOfInterest, CameraInfo
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from perceptive_stream.msg import BBox2D
from utils.utilsant import pose2Tmat, getTCCw
from utils.vecMatUtils import *
from utils import plucker
import numpy as np
import cv2
from collections import namedtuple
import time

from nav_msgs.msg import OccupancyGrid

Pos_error = namedtuple('Pos_error', 'x y z')
Rot_error = namedtuple('Rot_error', 'x y z')

class Projector:

    # Attributes
    #
    #   cam_pos_mat : 4x4 transform matrix
    #   K : camera matrix
    #   gnd_plane : Plucker's coordinates systeme based ground plane
    #   worked_ROI : table of tuple (pts_bbox, footprint_pts)
    #       pts_bbox : 
    #       footprint_pts : 

    def __init__(self, bboxes: BBox2D, gnd_plane = (vec4n(0, 0, 0), vec4n(1, 1, 0), vec4n(1, 0, 0)), step_grid = 0.2, grid_range = 75):
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
        # self.cam = vec4n(0, 0, 0)
        # self.cam = np.matmul(self.cam_pos_mat, self.cam)

        self.worked_ROI = []

        imgROI = RegionOfInterest(x_offset = 0, y_offset = 0, width = bboxes.cam_info.width, height = bboxes.cam_info.height)

        tic = time.process_time()
        (a, b) = self.compute_noisy_roi(roi=imgROI, cam_pos=self.cam_pos_mat, N_particles=30, pix_err=0.0, pos_err=Pos_error(0.1, 0.1, 0.01), rot_err=Rot_error(np.pi/180.0/5.0, np.pi/180.0/5.0, np.pi/180.0*3.0/2.0))
        # (a, b) = self.compute_roi(roi=imgROI, cam_pos=self.cam_pos_mat, log=True)
        b = np.minimum(b, len(ROIs))
        self.worked_ROI.append((a, b))

        if bboxes.header.frame_id.split('.')[0] == "Infra_camRGB":
            rospy.logwarn(a)

        for roi in ROIs: # For each ROI
            proj_roi = self.compute_noisy_roi(roi=roi, cam_pos=self.cam_pos_mat, N_particles=30, pix_err=5.0, pos_err=Pos_error(0.1, 0.1, 0.01), rot_err=Rot_error(np.pi/180.0/5.0, np.pi/180.0/5.0, np.pi/180.0*3.0/2.0))
            # proj_roi = self.compute_noisy_roi(roi=roi, cam_pos=self.cam_pos_mat, N_particles=1, pix_err=0.0, pos_err=Pos_error(0, 0, 0), rot_err=Rot_error(0, 0, 0))
            # proj_roi = self.compute_roi(roi=roi, cam_pos=self.cam_pos_mat)
            self.worked_ROI.append(proj_roi)
        toc = time.process_time()
        rospy.loginfo("Computed {} ROI of {} in {}s.".format(len(ROIs), bboxes.header.frame_id, toc-tic))

    def compute_noisy_roi(self, roi: RegionOfInterest, cam_pos, N_particles, pix_err=0.0, pos_err=Pos_error(0.0, 0.0, 0.0), rot_err=Rot_error(0.0, 0.0, 0.0)):
        (initial_pts_bbox, raw_map) = self.compute_roi(roi=roi, cam_pos=cam_pos)
        
        ox = np.random.normal(loc = roi.x_offset, scale = pix_err, size=N_particles)
        oy = np.random.normal(loc = roi.y_offset, scale = pix_err, size=N_particles)
        w = np.random.normal(loc = roi.width, scale = pix_err, size=N_particles)
        h = np.random.normal(loc = roi.height, scale = pix_err, size=N_particles)

        
        t = cam_pos[:3, 3]
        r = cam_pos[:3, :3]
        r = R.from_dcm(r)
        r_euler=R.as_euler(r, "xyz")

        noiseT = np.random.normal(loc=t, scale=[pos_err.x, pos_err.y, pos_err.z], size=(N_particles, 3))
        noiseR = np.random.normal(loc=r_euler, scale=[rot_err.x, rot_err.y, rot_err.z], size=(N_particles, 3))
        # rospy.logerr("noiseT :\n{}".format(noiseT))

        count: int
        count = 0

        for i in range(N_particles):
            noisyROI = RegionOfInterest(x_offset = int(ox[i]), y_offset = int(oy[i]), width = int(w[i]), height = int(h[i]))
            newT = np.identity(4)
            newT[:3, :3] = R.from_euler('xyz', noiseR[i]).as_dcm()
            newT[:3, 3] = noiseT[i]
            # rospy.logerr("Tcam :\n{}\nnewT :\n{}".format(cam_pos, newT))
            try:
                (_, new_map) = self.compute_roi(roi=noisyROI, cam_pos=newT)
            except OverflowError:
                rospy.logerr('Error with ROI:\n{}'.format(noisyROI))
            else:
                raw_map = np.add(raw_map, new_map)
                count = count + 1

        raw_map = np.divide(raw_map, count)
        raw_map = np.minimum(raw_map, 100)

        return (initial_pts_bbox, raw_map)

    def compute_roi(self, roi: RegionOfInterest, cam_pos, log=False):
        vertex = [] # extract each 
        pts_bbox = []
        cam = vec4n(0, 0, 0)
        cam = np.matmul(cam_pos, cam)
        pts_bbox.append([cam[i][0] for i in range(3)])
        
        vertex.append(vec3(roi.x_offset, roi.y_offset, 1))
        vertex.append(vec3(roi.x_offset + roi.width, roi.y_offset, 1))
        vertex.append(vec3(roi.x_offset + roi.width, roi.y_offset + roi.height, 1))
        vertex.append(vec3(roi.x_offset, roi.y_offset + roi.height, 1))
        for p in vertex:
            ps = np.matmul(self.K_inv, p) * (self.grid_range * 1.5)
            controlPS = np.matmul(self.K_inv, p) * 1.0
            pg = np.matmul(cam_pos, vec3tovec4(ps))
            controlPG = np.matmul(cam_pos, vec3tovec4(controlPS))
            L = plucker.line(cam, pg)
            fp_pt = plucker.interLinePlane(L, self.gnd_plane)

            if getNormVec4(fp_pt) > self.grid_range:
                # rospy.logerr("Out of Range.\nOut : {} -> {}".format(np.transpose(normVec4(fp_pt)), getNormVec4(fp_pt)))
                # rospy.logwarn(">>> {}\n{}".format(np.transpose(pg), roi))
                v = [normVec4(pg)[i][0] for i in range(3)]
                v[2] = 0
                pts_bbox.append(v)
            elif np.isnan(getNormVec4(fp_pt)):
                # rospy.logerr("Plucker error.\nOut : {} -> {}".format(np.transpose(normVec4(fp_pt)), getNormVec4(fp_pt)))
                # rospy.logwarn(">>> {}\n{}".format(np.transpose(pg), roi))
                v = [normVec4(pg)[i][0] for i in range(3)]
                v[2] = 0
                pts_bbox.append(v)
            elif pg[2][0] >= controlPG[2][0]:
                # rospy.logerr("Ray pointing the sky\nOut : {} -> {}".format(np.transpose(normVec4(fp_pt)), getNormVec4(fp_pt)))
                # rospy.logwarn(">>> {}\n{}".format(np.transpose(pg), roi))
                v = [normVec4(pg)[i][0] for i in range(3)]
                v[2] = 0
                pts_bbox.append(v)
            else:
                pts_bbox.append([normVec4(fp_pt)[i][0] for i in range(3)])

        raw_map = np.full((self.grid_size, self.grid_size), -1, dtype=np.float)

        pts = []
        for pt in pts_bbox[1:]:
            try:
                x = int(pt[0] / self.step_grid + (self.grid_size / 2.0))
                y = int(pt[1] / self.step_grid + (self.grid_size / 2.0))
            except OverflowError:
                rospy.logerr("Out : {}".format(pt))
            pts.append([x, y])
        
        cv2.fillPoly(raw_map, pts=[np.array(pts)], color=100)

        if log:
            rospy.logwarn(pts)
        

        return (pts_bbox, raw_map)


    def get_geometries(self):
        meshes = []
        # World ref
        # mesh_world_center = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

        # Camera ref
        mesh_camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh_camera.transform(self.cam_pos_mat)
        meshes.append(mesh_camera)

        for roi in self.worked_ROI:
            (pts_bbox, _) = roi
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
            # fp_points_td = []
            # for p in footprint_pts:
            #     fp_points_td.append([p[i][0] for i in range(3)])

            fp_lines_td = []
            footprint_pts = pts_bbox[1:]
            for i in range(len(footprint_pts)):
                fp_lines_td.append([i, (i+1)%len(footprint_pts)])

            colors_lines_fp = [[1, 0, 0] for i in range(len(fp_lines_td))]
            lineset_bbox_fp = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(footprint_pts),
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

        raw_map = np.full((self.grid_size, self.grid_size), -1, dtype=float)
        # raw_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        # rospy.logwarn("map size : {}".format(self.grid_size))

        for roi in self.worked_ROI:
            (_, map) = roi
            # pts = []
            raw_map = np.maximum(raw_map, map)


        rawGOL.data = raw_map.astype(np.int8).flatten().tolist()

        return rawGOL


