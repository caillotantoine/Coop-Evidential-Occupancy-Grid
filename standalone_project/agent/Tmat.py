import numpy as np
from copy import deepcopy
from typing import List
from scipy.spatial.transform import Rotation as R


class TMat:
    def __init__(self) -> None:
        self.tmat = np.identity(4)

    def __str__(self) -> str:
        return f"{self.tmat}"

    def get(self) -> np.ndarray:
        return self.tmat
    
    def set(self, mat) -> None:
        self.tmat = mat

    def reset(self) -> None:
        self.tmat = np.identity(4)

    def handinessLeft2Right(self):
        r = R.from_matrix(self.tmat[:3, :3])
        r_euler = R.as_euler(r, 'xyz')
        self.tmat[1, 3] = -self.tmat[1, 3]
        r_euler[0]=- r_euler[0]
        r_euler[2]=- r_euler[2]
        self.tmat[:3, :3] = R.from_euler('xyz', r_euler).as_matrix()

    # def translation(self, vector:vec4) -> None:
    #     self.tmat[0:4, 3] = vector.get().transpose()[0]

    def rotxMat(self, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        self.tmat = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def rotzMat(self, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        self.tmat = np.array([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def rotyMat(self, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        self.tmat = np.array([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def inv(self):
        self.tmat = np.linalg.inv(self.tmat)
    
    def __add__(self, other):
        from vector import vec4
        if type(other) == vec4:
            raise Exception('Prohibited.')
        res = self.tmat + other
        out = TMat()
        out.set(res)
        return out

    def __sub__(self, other):
        from vector import vec4
        if type(other) == vec4:
            raise Exception('Prohibited.')
        res = self.tmat + other
        out = TMat()
        out.set(res)
        return out

    def __mul__(self, other):
        from vector import vec4, vec3
        from bbox import Bbox3D
        if type(other) == int or type(other) == float:
            res = self.tmat * other
        elif type(other) == Bbox3D:
            pose3:vec3 = other.get_pose()
            pose4:vec4 = pose3.vec4()
            res:vec4 = self.tmat @ pose4.get()
            out = vec4(0, 0, 0)
            out.set(res)
            other.set_pose(out.nvec3())
            return other
        else:
            res = self.tmat @ other.get()
        if type(other) == vec4:
            out = vec4(0, 0, 0)
        else:
            out = TMat()
        out.set(res)
        return out


