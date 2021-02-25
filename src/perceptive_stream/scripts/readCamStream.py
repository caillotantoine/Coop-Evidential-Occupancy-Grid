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

def talker(camera_folder="/", info_folder="/", starting=0, sensor_ID="camera"):
    cnt = 2000

    bridge = CvBridge()
    img_msg = Img()
    
    
    pub = rospy.Publisher('camera/image', Img, queue_size=10)
    rospy.init_node('Camera_Publisher', anonymous=True)
    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        hello_str = "Hello world %s" % rospy.get_time()
        
        imagePath = '%s/%06d.png'%(camera_folder, starting+cnt)
        jsonPath = '%s/%06d.json'%(info_folder, starting+cnt)

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
            cameraMatrix = np.load('%s/cameraMatrix.npy'%camera_folder)
            img_msg.info.K = list(cameraMatrix.flatten())

            # Read the image
            img = cv.imread(imagePath)
            img_msg.image = bridge.cv2_to_imgmsg(img, "bgr8")

            # rospy.loginfo(hello_str)
            #rospy.loginfo("\n%s\n%s\n%s\n%s"%(np.array2string(cameraMatrix), imagePath, camera_Tr, camera_R))
            pub.publish(img_msg)
            cnt = cnt + 1
        else:
            cnt = 0
            rospy.loginfo("One of the following path doesn't exists. setting cnt to 0.\n\tPath 1: %s\n\tPath 2: %s"%(imagePath, jsonPath))
        rate.sleep()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--img_folder_path',
        help='Path to the folder from / to read the images (ex: /home/username/dataset/img/)'
    )
    argparser.add_argument(
        '--json_folder_path',
        help='Path to the folder from / to read the json files (ex: /home/username/dataset/json/)'
    )
    argparser.add_argument(
        '--starting',
        type=int,
        help='Number of the first file with the format %%06d.ext (ex: 46)'
    )
    argparser.add_argument(
        '--sensor_ID',
        default='camera',
        help='Sensor\'s ID to be stored in the header of the packet for each acquisition. Can be useful if several sensors stream in a single topic. (default: camera)'
    )
    args = argparser.parse_args()

    try:
        talker(camera_folder=args.img_folder_path, info_folder=args.json_folder_path, starting=args.starting, sensor_ID=args.sensor_ID)
    except rospy.ROSInterruptException:
        pass




