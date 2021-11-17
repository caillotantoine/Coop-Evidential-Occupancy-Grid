#!/usr/bin/env python3
# coding: utf-8
 
import numpy as np
from copy import deepcopy
from typing import List
from Tmat import TMat


class vec:
    def __init__(self, x: float, y: float, z: float=0.0, w:float = 0.0) -> None:
        self.vec = np.transpose(np.array([[x, y, z, w]]))

    def get(self) -> np.ndarray:
        return self.vec

    def set(self, vector):
        if type(vector) != np.ndarray:
            v = np.array(vector)
        else:
            v = vector

        if len(v.shape) == 2:
            (a, b) = v.shape
            if a == 1:
                self.vect = np.transpose(v)
            else:
                self.vec = v
        elif len(v.shape) == 1:
            self.vec = np.transpose(np.array([v]))

    def x(self):
        return self.vec[0,0]

    def y(self):
        return self.vec[1,0]

    def z(self):
        if type(self) == vec3 or type(self) == vec4:
            return self.vec[2,0]
        else:
            return None

    def w(self):
        if type(self) == vec4:
            return self.vec[3,0]
        else:
            return None

    def get_normalized(self):
        vsize = self.vec.size
        return self.vec / self.vec[vsize-1, 0]

    def normalize(self):
        self.vec = self.get_normalized()

    def get_norm(self):
        v = self.get_normalized()
        v = np.transpose(v)
        v = np.square(v)
        v = np.sum(v)
        return np.sqrt(v)

    def to_TMat(self) -> TMat:
        T = np.identity(4)
        T[0:vec.size, 3] = self.vec.transpose()[0]
        out = TMat()
        out.set(T)
        return out

    def __str__(self) -> str:
        out = "<"
        vsize = self.vec.size
        for i, n in enumerate(self.vec):
            if i != vsize-1:
                out = out + "%.3f, "%n
            else:
                out = out + "%.3f>"%n
        return out

    def __add__(self, other):
        out = deepcopy(self)
        if type(other) == type(self):
            o = other.get()
            for i, _ in enumerate(self.vec):
                out.vec[i, 0] = self.vec[i, 0] + o[i, 0]
        elif type(other) == int or type(other) == float:
            for i, _ in enumerate(self.vec):
                out.vec[i, 0] = self.vec[i, 0] + float(other)
        else:
            raise Exception(f"{type(self)} - {type(other)} was unexpected")
        return out

    def __sub__(self, other):
        out = deepcopy(self)
        if type(other) == type(self):
            o = other.get()
            for i, _ in enumerate(self.vec):
                out.vec[i, 0] = self.vec[i, 0] - o[i, 0]
        elif type(other) == int or type(other) == float:
            for i, _ in enumerate(self.vec):
                out.vec[i, 0] = self.vec[i, 0] - float(other)
        else:
            raise Exception(f"{type(self)} - {type(other)} was unexpected")
        return out

    def __mul__(self, other):
        out = deepcopy(self)
        if type(other) == type(self):
            out.set(np.dot(self.get(), other.get()))
        elif type(other) == int or type(other) == float:
            out.set(np.dot(self.get(), other))
        else:
            raise Exception(f"{type(self)} * {type(other)} was unexpected")
        return out

    def __truediv__(self, other):
        out = deepcopy(self)
        if type(other) == int or type(other) == float:
            out.set(self.get() / float(other))
        else:
            raise Exception(f"{type(self)} / {type(other)} was unexpected")
        return out

    def __floordiv__(self, other):
        out = deepcopy(self)
        if type(other) == int or type(other) == float:
            out.set(self.get() // float(other))
        else:
            raise Exception(f"{type(self)} // {type(other)} was unexpected")
        return out

    def __xor__(self, other):
        out = deepcopy(self)
        if type(self) == type(other):
            out.set( np.cross(self.get(), other.get()))
        else:
            raise Exception(f"{type(self)} ^ {type(other)} was unexpected")
        return out

    def __eq__(self, other):
        if type(other) == type(self):
            out = True
            o = other.get()
            for i, n in enumerate(self.vec):
                out = out and (self.vec[i, 0] == o[i, 0])
            return out
        elif type(other) == int or type(other) == float:
            return self.get_norm() == other
        else:
            raise Exception(f"{type(self)} * {type(other)} was unexpected")


class vec2(vec):
    def __init__(self, x:float, y:float) -> None:
        self.vec = np.transpose(np.array([[x, y]]))

    def vec3(self, z=0.0):
        x = self.vec[0,0]
        y = self.vec[1,0]
        return vec3(x, y, z)

    def vec4(self, z=0.0, w=1.0):
        x = self.vec[0,0]
        y = self.vec[1,0]
        return vec4(x, y, z, w)



class vec3(vec):
    def __init__(self, x:float, y:float, z:float=0.0) -> None:
        self.vec = np.transpose(np.array([[x, y, z]]))
    
    def vec4(self, w=1.0):
        x = self.vec[0,0]
        y = self.vec[1,0]
        z = self.vec[2,0]
        return vec4(x, y, z, w)

    def nvec2(self):
        v = self.get_normalized()
        x = v[0,0]
        y = v[1,0]
        return vec2(x, y)

    def vec2(self):
        x = self.vec[0,0]
        y = self.vec[1,0]
        return vec2(x, y)


class vec4(vec):
    def __init__(self, x:float, y:float, z:float, w: float=1.0) -> None:
        self.vec = np.transpose(np.array([[x, y, z, w]]))

    def nvec3(self) -> vec3:
        v = self.get_normalized()
        x = v[0,0]
        y = v[1,0]
        z = v[2,0]
        return vec3(x, y, z)

    def vec3(self) -> vec3:
        x = self.vec[0,0]
        y = self.vec[1,0]
        z = self.vec[2,0]
        return vec3(x, y, z)


if __name__ == "__main__":
    a = vec3(2.5, 3.0, 2.0)
    b = vec3(8.5, 5.3, 1.0)
    print(a)
    print(b)
    c = a + b
    print(a)
    print(c)
    print(c-b)
    print(c-3)
    # print(c^3)
    print(c == c)

    a = np.array([[2, 0, 0, 1], [0, 4, 0, 2], [0, 0, 6, 3], [0, 0, 0, 4]])
    mat = TMat()
    mat.set(a)
    print(mat)
    mat = mat * 2
    print(mat)
    b = vec4(2, 5, 8)
    print(b)
    print(mat * b)
    mat2 = TMat()
    mat2.set(np.array([[0.43231651186943054, 0.9017219543457031, 0.0, -16.7977352142334], [-0.9017219543457031, 0.43231651186943054, 0.0, 37.92546463012695], [0.0, -0.0, 1.0, 0.8238999247550964], [0.0, 0.0, 0.0, 1.0]]))
    print(mat2)
    mat2.handinessLeft2Right()
    print(mat2)
