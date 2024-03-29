#!/usr/bin/env python3
# coding: utf-8

import sys
from os import path
import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from perceptive_stream.msg import Img
from geometry_msgs.msg import Pose
import json
from pyquaternion import Quaternion
import argparse
from utils.utilsant import getTCCw, pose2Tmat, Tmat2pose

# Topics:
# image : the image of the camera
# info  : the info about the camera (matrice K, etc..)
# tf    : transformation camera in world

def talker():
    cnt = 0

    bridge = CvBridge()
    img_msg = Img()
    
    pub = rospy.Publisher('camera/image', Img, queue_size=100)
    rospy.init_node('Camera_Publisher', anonymous=True)
    

    path_to_npy = rospy.get_param('~npy_path')
    path_to_json = rospy.get_param('~json_path')
    path_to_img = rospy.get_param('~img_path')
    starting = rospy.get_param('starting')
    sensor_ID = rospy.get_param('~sensor_ID')
    rate = rospy.get_param('rate')
    rate = rospy.Rate(rate)

    rospy.loginfo("Path to npy : %s"%path_to_npy)
    rospy.loginfo("Path to img : %s"%path_to_img)
    rospy.loginfo("Path to json : %s"%path_to_json)
    rospy.loginfo("Starting point : %d"%starting)
    rospy.loginfo("Sensor ID : %s"%sensor_ID)

    cameraMatrix = np.load(path_to_npy)

    rospy.loginfo("Start streaming")
    while not rospy.is_shutdown():
        
        imagePath = path_to_img%(starting+cnt)
        jsonPath = path_to_json%(starting+cnt)

        if path.exists(jsonPath) and path.exists(imagePath):
            # Get the Sensor's position
            with open(jsonPath) as f:
                data = json.load(f)
            
            cameraT = np.array(data['sensor']['T_Mat'])

            img_msg.pose = Tmat2pose(cameraT, changeHandLeft2Right=True)

            # Get the Camera Matrix k
            img_msg.info.K = list(cameraMatrix.flatten())

            # Read the image
            img = cv.imread(imagePath)
            img_msg.image = bridge.cv2_to_imgmsg(img, "bgr8")

            img_msg.info.width = img_msg.image.width
            img_msg.info.height = img_msg.image.height


            # time to the ros server
            img_msg.header.stamp = rospy.get_rostime()
            img_msg.header.frame_id = "%s%d"%(sensor_ID, starting+cnt)
            pub.publish(img_msg)
            rate.sleep()

            cnt = cnt + 1
        else:
            rospy.loginfo("Reached cnt max : %d"%cnt)
            cnt = 0
            

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass




