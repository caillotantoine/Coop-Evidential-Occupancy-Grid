#!/usr/bin/env python3
# coding: utf-8
 
import numpy as np
from copy import deepcopy


class vec:
    def __init__(self) -> None:
        self.vec = np.transpose(np.array([[.0, .0]]))

    def get(self):
        return self.vec

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

    def to_TMat(self):
        T = np.identity(4)
        for i, n in enumerate(self.vec):
            T[i][3] = n
        return

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
        if type(other) == type(self):
            return np.dot(self.get(), other.get())
        elif type(other) == int or type(other) == float:
            return np.dot(self.get(), other)
        else:
            raise Exception(f"{type(self)} * {type(other)} was unexpected")

    def __div__(self, other):
        if type(other) == int or type(other) == float:
            return self.get() / float(other)
        else:
            raise Exception(f"{type(self)} / {type(other)} was unexpected")

    def __xor__(self, other):
        if type(self) == type(other):
            return np.cross(self.get(), other.get())
        else:
            raise Exception(f"{type(self)} ^ {type(other)} was unexpected")

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

class vec3(vec):
    def __init__(self, x:float, y:float, z:float=1.0) -> None:
        self.vec = np.transpose(np.array([[x, y, z]]))
    
    def toVec4(self):
        x = self.vec[0,0]
        y = self.vec[1,0]
        z = self.vec[2,0]
        return vec4(x, y, z)

class vec4(vec):
    def __init__(self, x:float, y:float, z:float, w: float=1.0) -> None:
        self.vec = np.transpose(np.array([[x, y, z, w]]))






def old_vec3(x, y, z):
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

def getNormVec4(v):
    a = normVec4(v)
    a = np.transpose(a)
    a = np.square(a)
    a = np.sum(a)
    return np.sqrt(a)


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