#!/usr/bin/env python3
# coding: utf-8
 
import numpy as np

def vec3(x, y, z):
    return np.transpose(np.array([[x, y, z]]))

def vec4n(x, y, z):
    return np.transpose(np.array([[x, y, z, 1.0]]))

def normVec4(v):
    return v / v[3][0]

def vec3tovec4(v):
    return np.transpose(np.array([[v[0][0], v[1][0], v[2][0], 1.0]]))

def vec2mat(v):
    T = np.identity(4)
    for i, n in enumerate(v):
        T[i][3] = n
    return T



def rotxMat(thetha):
    c = np.cos(thetha)
    s = np.sin(thetha)
    return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]])

def rotzMat(thetha):
    c = np.cos(thetha)
    s = np.sin(thetha)
    return np.array([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]])

def rotyMat(thetha):
    c = np.cos(thetha)
    s = np.sin(thetha)
    return np.array([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])