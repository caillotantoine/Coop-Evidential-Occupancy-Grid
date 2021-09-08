#!/usr/bin/env python3
# coding: utf-8

import sys
from os import path
import rospy
import json
from perceptive_stream.msg import BBox3D
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Vector3
from pyquaternion import Quaternion
import numpy as np

from utils.utilsant import getTCCw, pose2Tmat, Tmat2pose

def talker():
    bbox_msg = BBox3D()
    cnt = 0

    rospy.init_node('carGNDtruth_Publisher', anonymous=True)
    path_to_json = rospy.get_param('~json_path')
    starting_point = rospy.get_param('starting')
    rate = rospy.get_param('rate')
    vehicle_ID = rospy.get_param('~vehicle_ID')

    rospy.loginfo("Path to json : %s"%path_to_json)
    rospy.loginfo("Starting point : %d"%starting_point)
    rospy.loginfo("Vehicle ID : %s"%vehicle_ID)


    pub = rospy.Publisher('car/info', BBox3D, queue_size=100)
    rate = rospy.Rate(rate)


    rospy.loginfo('Start streaming')
    while not rospy.is_shutdown():
        jpath = path_to_json % (starting_point+cnt)
        # rospy.loginfo("Trying : %s"%jpath)
        if path.exists(jpath):
            with open(jpath) as f:
                data = json.load(f)
            
            center = Pose()
            center.position.x = data['vehicle']['BoundingBox']['loc']['x']
            center.position.y = data['vehicle']['BoundingBox']['loc']['y']
            center.position.z = data['vehicle']['BoundingBox']['loc']['z']

            size = Vector3()
            size.x = data['vehicle']['BoundingBox']['extent']['x']
            size.y = data['vehicle']['BoundingBox']['extent']['y']
            size.z = data['vehicle']['BoundingBox']['extent']['z']

            vehicle = Tmat2pose(data['vehicle']['T_Mat'], changeHandLeft2Right=True)

            bbox_msg.vehicle = vehicle
            bbox_msg.center = center
            bbox_msg.size = size
            bbox_msg.header.stamp = rospy.get_rostime()
            bbox_msg.header.frame_id = "%s%d"%(vehicle_ID, starting_point+cnt)

            pub.publish(bbox_msg)
            cnt = cnt + 1
            rate.sleep()
        else:
            rospy.loginfo("Reached cnt max : %d"%cnt)
            cnt = 0


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass


