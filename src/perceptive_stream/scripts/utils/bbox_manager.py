#!/usr/bin/env python3
# coding: utf-8

from numpy.core.numeric import Infinity
import rospy
import numpy as np
import cv2 as cv
from threading import Lock
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError
from pyquaternion import Quaternion
from perceptive_stream.msg import Img, BBox3D, BBox2D

from utils.utilsant import getTCCw, pose2Tmat

class BBoxManager:
    # cv_bridge : bridge openCV <-> ROS

    def __init__(self):

        # Creat an OpenCV bridge
        self.cv_bridge = CvBridge()



    def drawBoxes(self, image, bboxes):
        # take as input the image and the list of bounding box to draw
        # apply the draw to the image

        # convert the Image message to opoenCV format
        cv_img = self.cv_bridge.imgmsg_to_cv2(image.image, "bgr8")

        # information about the image an drawing parameters
        h, w, depth = cv_img.shape
        color = (0, 255, 0)
        thickness = 2


        for bbox in bboxes:
            #for each bounding box

            bbox_vert_world = self.placeBBoxInWorld(bbox) # place the 8 corner of the box in the world
            bbox_vert_cam = self.compute1bbox(image, bbox_vert_world)   # project the points in the camera plane

            points = [(int(x[0]), int(x[1])) for x in bbox_vert_cam] # we need tuple for the points

            rospy.loginfo("vertex :\n%s"%points)  

            # There can be no points if the bounding box is behind the camera. 
            if len(points) > 0:
                # draw each edges
                for idx in range(len(points)-1):
                    cv_img = cv.line(cv_img, points[idx], points[idx+1], color, thickness)
                cv_img = cv.line(cv_img, points[0], points[5], color, thickness)
                cv_img = cv.line(cv_img, points[1], points[6], color, thickness)
                cv_img = cv.line(cv_img, points[2], points[7], color, thickness)
                cv_img = cv.line(cv_img, points[4], points[7], color, thickness)
                cv_img = cv.line(cv_img, points[0], points[3], color, thickness)

        return self.cv_bridge.cv2_to_imgmsg(cv_img, "bgr8") # convert the openCV image to a ROS Image message

    def draw2DBoxes(self, image: Img, bboxes):
        # take as input the image and the list of bounding box to draw
        # apply the draw to the image

        # convert the Image message to opoenCV format
        cv_img = self.cv_bridge.imgmsg_to_cv2(image.image, "bgr8")

        # information about the image an drawing parameters
        h, w, depth = cv_img.shape
        color = (0, 255, 0)
        thickness = 2

        setOfBbox = []

        for bbox in bboxes:
            #for each bounding box

            bbox_vert_world = self.placeBBoxInWorld(bbox) # place the 8 corner of the box in the world
            (bbox_vert2D_cam, roi, valid) = self.compute2Dbbox(image, bbox_vert_world)

            # rospy.loginfo("Valid bbox: {}".format(valid))

            if valid and int(roi.x_offset) > 0 and int(roi.y_offset) > 0:
                if len(bbox_vert2D_cam) > 0:
                    setOfBbox.append(roi)

                for i in range(len(bbox_vert2D_cam)):
                    cv_img = cv.line(cv_img, bbox_vert2D_cam[i], bbox_vert2D_cam[(i+1)%len(bbox_vert2D_cam)], color, thickness)

        bbox_out = BBox2D()
        bbox_out.header = image.header
        bbox_out.cam_info = image.info
        bbox_out.cam_pose = image.pose
        bbox_out.roi = setOfBbox

        return (self.cv_bridge.cv2_to_imgmsg(cv_img, "bgr8"), bbox_out) # convert the openCV image to a ROS Image message

    
    def compute1bbox(self, image, bbox):
        # get intrinsec parameters (K) matrix of the camera.
        K = np.array(image.info.K)
        K = np.reshape(K, (3, 3))

        # Get the position of the camera
        T_WCw = pose2Tmat(image.pose) # cam -> world
        T_CwW = np.linalg.inv(T_WCw) # world -> cam

        T_CCw = getTCCw() # world system to camera system 

        T_CW = np.matmul(T_CCw, T_CwW) # world -> camera

        vertex = []
        for point in bbox:
            in_cam_space = np.matmul(T_CW, point) # place the point in camera space
            in_cam_space_cropped = in_cam_space[:3] # crop to get a 1x3 vector
            in_px_raw = np.matmul(K, in_cam_space_cropped) # project the point in the camera plane
            in_px_norm = in_px_raw / in_px_raw[2] # normalize the projection
            if in_px_raw[2] < 0.0: # if the point is behind the camera (z < 0), return an empty set of points
                return []
            vertex.append(in_px_norm) 
        return vertex

    def compute2Dbbox (self, image, bbox):
        bbox_vert_as3D = self.compute1bbox(image, bbox) # project the points in the camera plane
        points = [(int(x[0]), int(x[1])) for x in bbox_vert_as3D] # we need tuple for the points

        # Get the points of the 2D shape
        x_min = +10000
        x_max = -10000
        y_min = +10000
        y_max = -10000

        # Check if points are not too far from the image
        x_min_v = False
        x_max_v = False
        y_min_v = False
        y_max_v = False

        for pt in points:
            if pt[0] < x_min:
                x_min = pt[0]
                x_min_v = True
            if pt[1] < y_min:
                y_min = pt[1]
                y_min_v = True
            if pt[0] > x_max:
                x_max = pt[0]
                x_max_v = True
            if pt[1] > y_max:
                y_max = pt[1]
                y_max_v = True
        
        valid_lim = x_min_v and x_max_v and y_min_v and y_max_v

        is_on_image = False
        if (x_max > 0) and (x_max < image.image.width):
            is_on_image = is_on_image or True
        if (x_min > 0) and (x_min < image.image.width):
            is_on_image = is_on_image or True
        if (y_max > 0) and (y_max < image.image.height):
            is_on_image = is_on_image or True
        if (y_min > 0) and (y_min < image.image.height):
            is_on_image = is_on_image or True

        is_on_image = is_on_image and valid_lim

        roi = RegionOfInterest()
        roi.x_offset = x_min
        roi.y_offset = y_min
        roi.height = y_max - y_min
        roi.width = x_max - x_min

        points_out = []
        points_out.append((x_max, y_max))
        points_out.append((x_max, y_min))
        points_out.append((x_min, y_min))
        points_out.append((x_min, y_max))
        return (points_out, roi, is_on_image)


    def placeBBoxInWorld(self, bbox):

        # bounding box size
        l = bbox.size.x
        w = bbox.size.y
        h = bbox.size.z

        # Bounding box vertex (following CARLA description)
        vertex = []
        vertex.append(np.array([l, -w, -h, 1.0]))
        vertex.append(np.array([l, w, -h, 1.0]))
        vertex.append(np.array([-l, w, -h, 1.0]))
        vertex.append(np.array([-l, -w, -h, 1.0]))
        vertex.append(np.array([-l, -w, h, 1.0]))
        vertex.append(np.array([l, -w, h, 1.0]))
        vertex.append(np.array([l, w, h, 1.0]))
        vertex.append(np.array([-l, w, h, 1.0]))

        # Get Tmat to move the points around the car model at 0, 0, 0
        T_VB = np.identity(4)
        T_VB[0][3] = bbox.center.position.x
        T_VB[1][3] = bbox.center.position.y
        T_VB[2][3] = bbox.center.position.z

        # Apply to the vertex the transformation. The center of the Bounding Box and the vehicle location are different.
        for idx, v in enumerate(vertex):
            vertex[idx] = np.matmul(T_VB, v)
        
        # get Tmat vehicle -> world
        T_WV = pose2Tmat(bbox.vehicle)

        # Apply to the vertex the transformation. place every vertex in the world
        for idx, v in enumerate(vertex):
            vertex[idx] = np.matmul(T_WV, v)

        return vertex