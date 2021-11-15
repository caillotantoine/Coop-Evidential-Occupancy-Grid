import numpy as np
from vector import vec4

class Plucker:
    def get(self):
        return self.object

class plkrGND(Plucker):
    def __init__(self, A:vec4, B:vec4, C:vec4) -> None:
        M = np.concatenate((A.get().T, B.get().T, C.get().T), axis=0)
        M = np.transpose(M) 

        # Find the required determinants
        det234 = np.linalg.det(np.array([M[1], M[2], M[3]]))
        det134 = np.linalg.det(np.array([M[0], M[2], M[3]]))
        det124 = np.linalg.det(np.array([M[0], M[1], M[3]]))
        det123 = np.linalg.det(np.array([M[0], M[1], M[2]]))

        #create the vector of the plane
        self.object = np.transpose(np.array([[det234, -det134, det124, -det123]]))

class plkrLine(Plucker):
    def __init__(self, A:vec4, B:vec4) -> None:
        self.object = A.get()*B.get().T-B.get()*A.get().T
        


if __name__ == "__main__":
    A = vec4(0, 0, 0)
    B = vec4(1, 1, 0)
    C = vec4(1, 0, 0)
    plkrGND(A, B, C)