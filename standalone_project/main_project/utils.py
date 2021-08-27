#!/usr/bin/env python3
# coding: utf-8

from scipy.spatial.transform import Rotation as R
import numpy as np

def changeHand(mat):
    T = mat[:3, 3:4]
    r = R.from_dcm(mat[:3, :3])
    r_euler = R.as_euler(r, 'xyz')
    T[1] = -T[1]
    r_euler[0]=- r_euler[0]
    r_euler[2]=- r_euler[2]
    r_out = R.from_euler('xyz', r_euler).as_dcm()
    out = np.identity(4)
    out[:3, 3:4] = T
    out[:3, :3] = r_out
    return out

def getTCCw():
    return np.array([[0.0,   -1.0,   0.0,   0.0,], [0.0,   0.0,  -1.0,   0.0,], [1.0,   0.0,   0.0,   0.0,], [0.0,   0.0,   0.0,   1.0,]])
    # matrix to change from world space to camera space 