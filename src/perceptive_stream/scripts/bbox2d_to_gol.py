#!/usr/bin/env python3
# coding: utf-8

from utils.projector import Projector
import rospy
import open3d as o3d
import numpy as np
import time
from threading import Lock
from scipy.spatial.transform import Rotation as R
from utils import plucker 
import copy
from nav_msgs.msg import OccupancyGrid

# from sensor_msgs.msg import CameraInfo, RegionOfInterest
from perceptive_stream.msg import BBox2D

map_size = 70
mesh_world_center = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])


class BBox2D_Proj:
    def __init__(self):
        rospy.init_node("bbox_to_gol", anonymous=True)
        rospy.Subscriber('projector/bbox2d', BBox2D, self.callback_bbox)
        self.go_pub = rospy.Publisher('projector/GOL', OccupancyGrid, queue_size=10)
        try: 
            do_vis = rospy.get_param('~o3d')
            self.do_vis = do_vis
        except KeyError:
            self.do_vis = False

        if self.do_vis:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=rospy.get_name(), width=800, height=600, left=50, top=50, visible=True)

        self.is_modified = True
        self.list_geometries = []
        self.mutex_geometries = Lock()
        modif = False
        geometries = []
        
        if self.do_vis:
            (line_setX, line_setY) = self.create_ground_grid()

        if self.do_vis:
            vis.add_geometry(mesh_world_center)
            vis.add_geometry(line_setX)
            vis.add_geometry(line_setY)
        while not rospy.is_shutdown():
            if self.do_vis:
                self.mutex_geometries.acquire()
                try:
                    if self.is_modified:
                        new_geometries = copy.deepcopy(self.list_geometries) 
                        modif = copy.deepcopy(self.is_modified)
                        self.is_modified = False
                finally:
                    self.mutex_geometries.release()

                if modif:
                    modif = False
                    for geo in geometries:
                        vis.remove_geometry(geo)
                        pass
                    geometries.clear()
                    geometries = new_geometries
                    for geo in geometries:
                        vis.add_geometry(geo)

                vis.poll_events()
                vis.update_renderer()
        if self.do_vis:
            vis.destroy_window()

    def callback_bbox(self, data: BBox2D):
        rospy.logwarn("GOL publisher : {}\n".format(data.header.frame_id))
        proj = Projector(data)
        if self.do_vis:
            self.mutex_geometries.acquire(blocking=True)
            try:
                if data.header.frame_id.split('.')[0] == "Infra_camRGB":
                    self.is_modified = True
                    self.list_geometries = proj.get_geometries()
            finally:
                self.mutex_geometries.release()
        gol = proj.get_occupGrid()
        gol.header = data.header
        self.go_pub.publish(gol)


    def create_ground_grid(self):
        x_col = [0.5, 0.5, 0.5]
        y_col = [0.5, 0.5, 0.5]
        pointsX = []
        pointsY = []
        lineX = []
        lineY = []

        for i in range(-map_size, map_size, 1):
            pointsX.append([i, -map_size, 0])
            pointsX.append([i, map_size, 0])


        for i in range(0, len(pointsX), 2):
            lineX.append([i, i+1])

        for i in range(-map_size, map_size, 1):
            pointsY.append([-map_size, i, 0])
            pointsY.append([map_size, i, 0])

        for i in range(0, len(pointsX), 2):
            lineY.append([i, i+1])

        colorsX = [x_col for i in range(len(lineX))]
        line_setX = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pointsX),
            lines=o3d.utility.Vector2iVector(lineX)
        )
        line_setX.colors = o3d.utility.Vector3dVector(colorsX)

        colorsY = [y_col for i in range(len(lineY))]
        line_setY = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pointsY),
            lines=o3d.utility.Vector2iVector(lineY)
        )
        line_setY.colors = o3d.utility.Vector3dVector(colorsY)
        return (line_setX, line_setY)

if __name__ == '__main__':
    # renderer = Thread(target=gui_pipeline, args=(1,))
    # renderer.start()
    proj_node = BBox2D_Proj()