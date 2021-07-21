#!/usr/bin/env python3
# coding: utf-8

# input:
#           Image
#           BBox 3D
#
# output:
#           GOL (ego vehicle - centered)

import rospy
import numpy as np
import cv2 as cv
from threading import Lock
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError
from pyquaternion import Quaternion
from perceptive_stream.msg import Img, BBox3D, BBox2D

from utils.bbox_manager import BBoxManager
from utils.queue_manager import QueueROSmsgs

class BBoxExtractor:
    #pub : the publisher
    #pub_mutex : the mutex to avoid publishing in the same time. 

    def __init__(self):
        # init ROS 
        rospy.init_node("bbox_extractor", anonymous=True)
        queue_size = rospy.get_param('~queue_size')

        # create a publisher
        self.pub = rospy.Publisher('projector/img', Image, queue_size=10)
        self.pub_mutex = Lock()
        self.pub_bbox = rospy.Publisher('projector/bbox2d', BBox2D, queue_size=10)
        # self.pub_bbox_mutex = Lock()

        # Creat an OpenCV bridge
        self.cv_bridge = CvBridge()

        # Create a queue for the bounding box. Used to find a matching Bbox with an image
        self.Bbox_queue = QueueROSmsgs(queue_size, debug=True)

        self.bboxMgr= BBoxManager()

        rospy.Subscriber('camera/image', Img, self.callback_img)
        rospy.Subscriber('car/info', BBox3D, self.callback_bbox)

        rospy.spin()
    
    def callback_img(self, data):
        # Display and record the frame ID number (used to synchronize)
        img_id = data.header.frame_id.split('.')[1]
        rospy.loginfo("Received img : %s", img_id)

        # There can be several bounding box (one per vehicles)
        listOfBBox = []
        bboxFound = True
        while(bboxFound):
            bboxIdx = self.Bbox_queue.searchFirst(".%s"%img_id) # Looking for corresponding bounding box with this image
            if bboxIdx == -1:
                bboxFound = False 
                break # there is no bounding box anymore, let's get out of the loop
            listOfBBox.append(self.Bbox_queue.pop(bboxIdx)) # remove the corresponding bounding box from the que and add it to the local list

        img_msg = Image()

        if len(listOfBBox) > 0:
            # If there are some bounding box, draw them on the image
            # img_msg = self.drawBoxes(data, listOfBBox)
            (img_msg, bbox2D) = self.bboxMgr.draw2DBoxes(data, listOfBBox)
            for bbox in listOfBBox:
                rospy.loginfo("position : X %3.2fm   \tY %3.2fm"%(bbox.vehicle.position.x, bbox.vehicle.position.y))
            for bbox in bbox2D.roi:
                rospy.loginfo("BBox : ")
                rospy.loginfo("    x: %d  y: %d  w: %d  h: %d"%(bbox.x_offset, bbox.y_offset, bbox.width, bbox.height))
        else:
            # if there is no bounding box, just pipe the image
            img_msg = data.image

        img_msg.header = data.header

        self.pub_mutex.acquire()        # Is it mendatory?
        try:
            self.pub.publish(img_msg) # publish
            if len(listOfBBox) > 0:
                self.pub_bbox.publish(bbox2D)
        finally:
            self.pub_mutex.release()
        rospy.loginfo("End process img : %s", img_id)

    def callback_bbox(self, data):
        self.Bbox_queue.add(data) # just add the read bounding box to the queue


if __name__ == '__main__':
    proj_node = BBoxExtractor()