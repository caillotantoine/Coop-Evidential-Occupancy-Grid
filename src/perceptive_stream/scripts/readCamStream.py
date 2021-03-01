#!/usr/bin/env python

import sys
from os import path
import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from perceptive_stream.msg import Img
from geometry_msgs.msg import Transform
import json
from pyquaternion import Quaternion
import argparse

# Topics:
# image : the image of the camera
# info  : the info about the camera (matrice K, etc..)
# tf    : transformation camera in world

def talker():
    cnt = 0

    bridge = CvBridge()
    img_msg = Img()
    
    pub = rospy.Publisher('camera/image', Img, queue_size=10)
    rospy.init_node('Camera_Publisher', anonymous=True)
    rate = rospy.Rate(20)

    path_to_npy = rospy.get_param('~npy_path')
    path_to_json = rospy.get_param('~json_path')
    path_to_img = rospy.get_param('~img_path')
    starting = rospy.get_param('starting')
    sensor_ID = rospy.get_param('~sensor_ID')

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
            camera_t = cameraT[:3, 3:4]
            camera_R = Quaternion(matrix=cameraT[:3, :3])

            camera_Tr = Transform()
            camera_Tr.translation.x = camera_t.flatten()[0]
            camera_Tr.translation.y = camera_t.flatten()[1]
            camera_Tr.translation.z = camera_t.flatten()[2]
            camera_Tr.rotation.w = camera_R.w
            camera_Tr.rotation.x = camera_R.x
            camera_Tr.rotation.y = camera_R.y
            camera_Tr.rotation.z = camera_R.z

            img_msg.transform = camera_Tr

            # Get the Camera Matrix k
            img_msg.info.K = list(cameraMatrix.flatten())

            # Read the image
            img = cv.imread(imagePath)
            img_msg.image = bridge.cv2_to_imgmsg(img, "bgr8")

            # time to the ros server
            img_msg.header.stamp = rospy.get_rostime()
            img_msg.header.frame_id = "%s_%d"%(sensor_ID, starting+cnt)
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




