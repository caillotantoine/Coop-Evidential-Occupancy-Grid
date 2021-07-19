
import numpy as np

def plane(A, B, C):
    # Creation of a plane in Plucker coordinates as defined at 
    # page 67 in Multiple View Geometry in computer vision
    # by R. Hartley and A. Zisserman

    M = np.concatenate((A.T, B.T, C.T), axis=0) # create the matrix M
    M = np.transpose(M) 

    # Find the required determinants
    det234 = np.linalg.det(np.array([M[1], M[2], M[3]]))
    det134 = np.linalg.det(np.array([M[0], M[2], M[3]]))
    det124 = np.linalg.det(np.array([M[0], M[1], M[3]]))
    det123 = np.linalg.det(np.array([M[0], M[1], M[2]]))

    #create the vector of the plane
    P = np.transpose(np.array([[det234, -det134, det124, -det123]]))

    return P

def line(A, B):
    # Creation of a Line from 2 points in Plucker coordinates as defined at 
    # page 71 in Multiple View Geometry in computer vision
    # by R. Hartley and A. Zisserman
    L = A*B.T-B*A.T
    return L

def interLinePlane(L, P):
    # Finding the point at the intersection of the line L and the plane P as defined at 
    # page 72 in Multiple View Geometry in computer vision
    # by R. Hartley and A. Zisserman
    return np.matmul(L, P)