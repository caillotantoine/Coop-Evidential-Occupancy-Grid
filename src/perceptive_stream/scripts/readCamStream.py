#!/usr/bin/env python

import sys
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

# Topics:
# image : the image of the camera
# info  : the info about the camera (matrice K, etc..)
# tf    : transformation camera in world

def talker():
    camera_folder = '/home/caillot/Bureau/Dataset_Test_1/Infra/cameraRGB/'
    info_folder = '/home/caillot/Bureau/Dataset_Test_1/Infra/sensorInfo/'
    starting = 46
    cnt = 0

    imagePath = '%s/%06d.png'%(camera_folder, starting+cnt)
    jsonPath = '%s/%06d.json'%(info_folder, starting+cnt)

    bridge = CvBridge()



    


    
    

    img_msg = Img()
    
    


    pub = rospy.Publisher('camera/image', Img, queue_size=10)
    rospy.init_node('Camera_Publisher', anonymous=True)
    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        hello_str = "Hello world %s" % rospy.get_time()
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
        rospy.loginfo("\n%s\n%s\n%s\n%s"%(np.array2string(cameraMatrix), imagePath, camera_Tr, camera_R))
        pub.publish(img_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass




