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
import utilsant as toto

print(toto.getTCCw())