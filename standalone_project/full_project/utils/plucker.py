import numpy as np
from utils.vector import vec4

class Plucker:
    def get(self):
        return self.object

    def set(self, obj):
        self.object = obj

    # Finding the point at the intersection of the line L and the plane P as defined at 
    # page 72 in Multiple View Geometry in computer vision
    # by R. Hartley and A. Zisserman
    def inter_plane_line(self, Plane, Line) -> vec4:
        L = Line.get()
        P = Plane.get()
        res = np.matmul(L, P)
        out = vec4(0, 0, 0)
        out.set(res)
        return out


class plkrPlane(Plucker):
    # Creation of a plane in Plucker coordinates as defined at 
    # page 67 in Multiple View Geometry in computer vision
    # by R. Hartley and A. Zisserman
    def __init__(self, A:vec4 = vec4(0, 0, 0), B:vec4 = vec4(1, 0, 0), C:vec4 = vec4(0, 1, 0)) -> None:
        M = np.concatenate((A.get().T, B.get().T, C.get().T), axis=0)
        M = np.transpose(M) 

        # Find the required determinants
        det234 = np.linalg.det(np.array([M[1], M[2], M[3]]))
        det134 = np.linalg.det(np.array([M[0], M[2], M[3]]))
        det124 = np.linalg.det(np.array([M[0], M[1], M[3]]))
        det123 = np.linalg.det(np.array([M[0], M[1], M[2]]))

        #create the vector of the plane
        self.object = np.transpose(np.array([[det234, -det134, det124, -det123]]))

    def intersect(self, other):
        if type(other) == plkrLine:
            return self.inter_plane_line(self, other)
        else:
            raise Exception("This intersection is not developped yet.")

class plkrLine(Plucker):
    # Creation of a Line from 2 points in Plucker coordinates as defined at 
    # page 71 in Multiple View Geometry in computer vision
    # by R. Hartley and A. Zisserman
    def __init__(self, A:vec4 = vec4(0,0,0), B:vec4 = vec4(0,0,1)) -> None:
        self.object = A.get()*B.get().T-B.get()*A.get().T

    def intersect(self, other):
        if type(other) == plkrPlane:
            return self.inter_plane_line(other, self)
        else:
            raise Exception("This intersection is not developped yet.")
        


if __name__ == "__main__":
    A = vec4(0, 0, 0)
    B = vec4(1, 1, 0)
    C = vec4(1, 0, 0)
    gnd = plkrPlane(A, B, C)
    print(gnd.get())


    D = vec4(1, 2, 1)
    E = vec4(1, 2, 3)
    line = plkrLine(D, E)
    print(line.get())

    print(line.intersect(gnd).get_normalized())