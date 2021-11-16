import numpy as np
from vector import vec2, vec3, vec4
from typing import List

class Bbox:
    def __init__(self, pose, size, label:str = "") -> None:
        self.pose = pose
        self.size = size
        self.label = label
        

    def get_pose(self):
        return self.pose
    
    def get_size(self):
        return self.size

    def get_label(self):
        return self.label

class Bbox3D(Bbox):
    def __init__(self, pose:vec3, size:vec3, label:str) -> None:
        super().__init__(pose, size, label)

    def set_from_pts(self, points: List[vec3]):
        lpts = []
        for v in points:
            lpts.append(np.transpose(v.get())[0])
        lpts = np.array(lpts)
        ox = np.amin(lpts[:, 0])
        sx = np.amax(lpts[:, 0]) - np.amin(lpts[:, 0])
        oy = np.amin(lpts[:, 1])
        sy = np.amax(lpts[:, 1]) - np.amin(lpts[:, 1])
        oz = np.amin(lpts[:, 2])
        sz = np.amax(lpts[:, 2]) - np.amin(lpts[:, 2])
        self.pose = vec3(ox, oy,  oz)
        self.size = vec3(sx, sy,  sz)

    def get_pts(self) -> List[vec3]:
        lpts:List[vec3] = []
        lpts.append(vec3(self.pose.x(), self.pose.y(), self.pose.z()))
        lpts.append(vec3(self.pose.x() + self.size.x(), self.pose.y(), self.pose.z()))
        lpts.append(vec3(self.pose.x() + self.size.x(), self.pose.y() + self.size.y(), self.pose.z()))
        lpts.append(vec3(self.pose.x(), self.pose.y() + self.size.y(), self.pose.z()))
        lpts.append(vec3(self.pose.x(), self.pose.y() + self.size.y(), self.pose.z() + self.size.z()))
        lpts.append(vec3(self.pose.x() + self.size.x(), self.pose.y() + self.size.y(), self.pose.z() + self.size.z()))
        lpts.append(vec3(self.pose.x() + self.size.x(), self.pose.y(), self.pose.z() + self.size.z()))
        lpts.append(vec3(self.pose.x(), self.pose.y(), self.pose.z() + self.size.z()))
        return lpts

class Bbox2D(Bbox):
    def __init__(self, pose:vec2, size:vec2, label:str) -> None:
        super().__init__(pose, size, label)

    def set_from_pts(self, points: List[vec2]):
        lpts = []
        for v in points:
            lpts.append(np.transpose(v.get())[0])
        lpts = np.array(lpts)
        ox = np.amin(lpts[:, 0])
        sx = np.amax(lpts[:, 0]) - np.amin(lpts[:, 0])
        oy = np.amin(lpts[:, 1])
        sy = np.amax(lpts[:, 1]) - np.amin(lpts[:, 1])
        self.pose = vec2(ox, oy)
        self.size = vec2(sx, sy)

    def get_pts(self) -> List[vec2]:
        lpts:List[vec2] = []
        lpts.append(vec2(self.pose.x(), self.pose.y()))
        lpts.append(vec2(self.pose.x() + self.size.x(), self.pose.y()))
        lpts.append(vec2(self.pose.x() + self.size.x(), self.pose.y() + self.size.y()))
        lpts.append(vec2(self.pose.x(), self.pose.y() + self.size.y()))
        return lpts


if __name__ == "__main__":
    pts = []
    pts.append(vec2(2, 2))
    pts.append(vec2(2, 8))
    pts.append(vec2(8, 8))
    pts.append(vec2(8, 2))
    b = Bbox2D(None, None)
    b.set_from_pts(pts)
    print([f"{x}" for x in b.get_pts()])