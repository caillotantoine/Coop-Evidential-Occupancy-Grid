#!/usr/bin/env python3
# coding: utf-8

import rospy
import numpy as np
import cv2 as cv
from threading import Lock
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pyquaternion import Quaternion
from perceptive_stream.msg import Img, BBox3D

from bbox_manager import BBoxProj


if __name__ == '__main__':
    proj_node = BBoxProj()