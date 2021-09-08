#!/usr/bin/env python3
# coding: utf-8

# from codecs import xmlcharrefreplace_errors
# from typing import NamedTuple
# from numpy.core.numerictypes import obj2sctype
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
import multiprocessing as mp
import copy

from nav_msgs.msg import OccupancyGrid

Pos_error = namedtuple('Pos_error', 'x y z')
Rot_error = namedtuple('Rot_error', 'x y z')

def compute_roi_fast(args):
    tic = time.process_time()
    (roi, cam_pos, K_inv, grid, gnd_plane) = copy.deepcopy(args)
    
    vertex = [] # extract each 
    pts_bbox = []
    cam = vec4n(0, 0, 0)
    cam = np.matmul(cam_pos, cam)
    pts_bbox.append([cam[i][0] for i in range(3)])
    
    vertex.append(vec3(roi.x_offset, roi.y_offset, 1))
    vertex.append(vec3(roi.x_offset + roi.width, roi.y_offset, 1))
    vertex.append(vec3(roi.x_offset + roi.width, roi.y_offset + roi.height, 1))
    vertex.append(vec3(roi.x_offset, roi.y_offset + roi.height, 1))
    (grid_range, grid_size, step_grid) = grid

    for p in vertex:
        ps = np.matmul(K_inv, p) * (grid_range * 1.5)
        controlPS = np.matmul(K_inv, p) * 1.0
        pg = np.matmul(cam_pos, vec3tovec4(ps))
        controlPG = np.matmul(cam_pos, vec3tovec4(controlPS))
        L = plucker.line(cam, pg)
        fp_pt = plucker.interLinePlane(L, gnd_plane)
        
        if np.isnan(getNormVec4(fp_pt)) or pg[2][0] >= controlPG[2][0]:
            v = [normVec4(pg)[i][0] for i in range(3)]
            v[2] = 0
            pts_bbox.append(v)
        else:
            pts_bbox.append([normVec4(fp_pt)[i][0] for i in range(3)])

    # raw_map = np.full((grid_size, grid_size), -1, dtype=np.float)

    pts = []
    for pt in pts_bbox[1:]:
        x = int(pt[0] / step_grid + (grid_size / 2.0))
        y = int(pt[1] / step_grid + (grid_size / 2.0))
        pts.append([x, y])

    
    # cv2.fillPoly(raw_map, pts=[np.array(pts)], color=100)
    toc = time.process_time()
    # rospy.logwarn("Computed 1 particles in {}s".format(toc-tic))
    return pts #raw_map

def draw_fast(args):
    (buf, cam_map, new_map) = args
    cv2.fillPoly(buf, pts=[np.array(cam_map)], color=0)
    if new_map != None:
        cv2.fillPoly(buf, pts=[np.array(new_map)], color=100)
    return buf

class Projector:

    # Attributes
    #
    #   cam_pos_mat : 4x4 transform matrix
    #   K : camera matrix
    #   gnd_plane : Plucker's coordinates systeme based ground plane
    #   worked_ROI : table of tuple (pts_bbox, footprint_pts)
    #       pts_bbox : 
    #       footprint_pts : 

    def __init__(self, bboxes: BBox2D, gnd_plane = (vec4n(0, 0, 0), vec4n(1, 1, 0), vec4n(1, 0, 0)), step_grid = 0.2, grid_range = 75, N_part=1, px_noise=0.0, trans_noise=(0.0, 0.0), rot_noise=(0.0, 0.0)):
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
        # rospy.logerr("N part = {}".format(N_part))
        

        tic = time.process_time()
        # (a, b) = self.compute_noisy_roi(roi=imgROI, cam_pos=self.cam_pos_mat, N_particles=N_part, pix_err=0, pos_err=Pos_error(0.1, 0.1, 0.01), rot_err=Rot_error(np.pi/180.0/5.0, np.pi/180.0/5.0, np.pi/180.0*3.0/2.0))
        # # (a, b) = self.compute_roi(roi=imgROI, cam_pos=self.cam_pos_mat, log=True)
        # b = np.minimum(b, len(ROIs))
        # self.worked_ROI.append((a, b))
        self.pool = mp.Pool(mp.cpu_count()-2)
        if len(ROIs) == 0:
            proj_roi = self.compute_noisy_roi(roi=None, cam_pos=self.cam_pos_mat, cam_roi=imgROI, N_particles=N_part, pix_err=px_noise) # No position error for the infrastructure
            self.worked_ROI.append(proj_roi)
        
        # rospy.logerr("CPU count %d"%mp.cpu_count())
        for roi in ROIs: # For each ROI
            if bboxes.header.frame_id.split('.')[0] == "Infra_camRGB":
                proj_roi = self.compute_noisy_roi(roi=roi, cam_pos=self.cam_pos_mat, cam_roi=imgROI, N_particles=N_part, pix_err=px_noise) # No position error for the infrastructure
            else:
               proj_roi = self.compute_noisy_roi(roi=roi, cam_pos=self.cam_pos_mat, cam_roi=imgROI, N_particles=N_part, pix_err=px_noise, pos_err=Pos_error(trans_noise[0], trans_noise[0], trans_noise[1]), rot_err=Rot_error(np.pi/180.0*rot_noise[0], np.pi/180.0*rot_noise[0], np.pi/180.0*rot_noise[1])) 
            self.worked_ROI.append(proj_roi)
        self.pool.close()
        toc = time.process_time()
        rospy.loginfo("Computed {} ROI of {} in {}s.".format(len(ROIs)+1, bboxes.header.frame_id, toc-tic))

    def compute_noisy_roi(self, roi: RegionOfInterest, cam_pos, cam_roi, N_particles, pix_err=0.0, pos_err=Pos_error(0.0, 0.0, 0.0), rot_err=Rot_error(0.0, 0.0, 0.0)): 
        grid = (self.grid_range, self.grid_size, self.step_grid)
        t = cam_pos[:3, 3]
        r = cam_pos[:3, :3]
        r = R.from_dcm(r)
        r_euler=R.as_euler(r, "xyz")
        noiseT = np.random.normal(loc=t, scale=[pos_err.x, pos_err.y, pos_err.z], size=(N_particles, 3))
        noiseR = np.random.normal(loc=r_euler, scale=[rot_err.x, rot_err.y, rot_err.z], size=(N_particles, 3))

        noised_tools = []
        noised_tools_A = []
        noised_tools_B = []

        if roi != None:
            ox = np.random.normal(loc = roi.x_offset, scale = pix_err, size=N_particles)
            oy = np.random.normal(loc = roi.y_offset, scale = pix_err, size=N_particles)
            w = np.random.normal(loc = roi.width, scale = pix_err, size=N_particles)
            h = np.random.normal(loc = roi.height, scale = pix_err, size=N_particles)
            
        for i in range(N_particles):
            if roi != None:
                noisyROI = RegionOfInterest(x_offset = int(ox[i]), y_offset = int(oy[i]), width = int(w[i]), height = int(h[i]))
            newT = np.identity(4)
            newT[:3, :3] = R.from_euler('xyz', noiseR[i]).as_dcm()
            newT[:3, 3] = noiseT[i]
            # noised_tools.append((noisyROI, cam_roi, newT))
            if roi != None:
                noised_tools_A.append((noisyROI, newT, self.K_inv, grid, self.gnd_plane))
            noised_tools_B.append((cam_roi, newT, self.K_inv, grid, self.gnd_plane))
        toc = time.process_time()
        # rospy.logerr("Generate particles in {}s".format(toc-tic))

        tic = time.process_time()
        maps = []       
        
        cam_map = self.pool.map(compute_roi_fast, noised_tools_B)
        if roi != None:
            new_map = self.pool.map(compute_roi_fast, noised_tools_A)
        else:
            new_map = [None for _ in range(len(cam_map))]
        toc = time.process_time()
        # rospy.logerr("Compute Pools in {}s".format(toc-tic))

        tic = time.process_time()
        if roi != None:
            (initial_pts_bbox, _) = self.compute_roi(roi=roi, cam_pos=cam_pos)
        else:
            (initial_pts_bbox, _) = self.compute_roi(roi=cam_roi, cam_pos=cam_pos)
        buf = np.full((self.grid_size, self.grid_size), -1, dtype=np.float)
        vals = []
        for i in range(len(cam_map)):
            vals.append((copy.deepcopy(buf), cam_map[i], new_map[i]))

        outs = []
        for a in vals:
            outs.append(draw_fast(a))   
        toc = time.process_time()
        # rospy.logerr("Compute maps in {}s".format(toc-tic))
        
        
        tic = time.process_time()
        smap = np.sum(outs, axis=0)
        smap = np.divide(smap, N_particles)
        smap = np.minimum(smap, 100)
        toc = time.process_time()
        # rospy.logerr("Merge particules in {}s".format(toc-tic))

        # rospy.logerr("Shape : {}".format(np.shape(raw_map)))

        return (initial_pts_bbox, smap)

    def compute_noisy_roi_old(self, roi: RegionOfInterest, cam_pos, cam_roi, N_particles, pix_err=0.0, pos_err=Pos_error(0.0, 0.0, 0.0), rot_err=Rot_error(0.0, 0.0, 0.0)):
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


        for i in range(N_particles):
            noisyROI = RegionOfInterest(x_offset = int(ox[i]), y_offset = int(oy[i]), width = int(w[i]), height = int(h[i]))
            newT = np.identity(4)
            newT[:3, :3] = R.from_euler('xyz', noiseR[i]).as_dcm()
            newT[:3, 3] = noiseT[i]

            (_, cam_map) = self.compute_roi(roi=cam_roi, cam_pos=newT)
            cam_map = np.minimum(cam_map, 0)
            (_, new_map) = self.compute_roi(roi=noisyROI, cam_pos=newT)
            merged_map = np.maximum(cam_map, new_map)
            raw_map = np.add(raw_map, merged_map)

        raw_map = np.divide(raw_map, N_particles)
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
            
            if np.isnan(getNormVec4(fp_pt)) or pg[2][0] >= controlPG[2][0]:
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

        # if log:
        #     rospy.logwarn(pts)
        

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


