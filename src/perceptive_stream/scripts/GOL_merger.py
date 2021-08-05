#!/usr/bin/env python3
# coding: utf-8

from numpy.testing._private.utils import KnownFailureException
import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
from utils.DST import DST
import copy

class GOmerger:

    def __init__(self):
        rospy.init_node("GO_merger", anonymous=True)
        rospy.Subscriber('projector/GOL', OccupancyGrid, self.callback)
        self.pub = rospy.Publisher('merger/GO', OccupancyGrid, queue_size=10)
        self.usersGOL = []

        self.fig = None
        


        rospy.spin()

    def callback (self, data: OccupancyGrid):
        # rospy.logwarn("GOL Merger =============================\n{}\n".format(data.header))
        GOLfound = False
        maps_instock = []
        for m in self.usersGOL:
            maps_instock.append(m.header.frame_id)
        rospy.logerr("New map received : {}\nOld ones : {}".format(data.header.frame_id, maps_instock))

        
        for i,gol in enumerate(self.usersGOL):
            if data.header.frame_id.split('.')[0] == gol.header.frame_id.split('.')[0]:
                rospy.logerr("Found! replace!")
                self.usersGOL[i] = data
                GOLfound = True
                break
        if not GOLfound:
            self.usersGOL.append(data)

        if self.fig is None:
            self.fig, self.axes = plt.subplots(2, 3)
            self.fig.suptitle("Occupancy grids")

        for i,gol in enumerate(self.usersGOL):
            # rospy.logerr("Show : {} on [{}, {}] (i = {})".format(gol.header.frame_id, i//3, i%3, i))
            self.axes[i//3, i%3].imshow(np.add(gol.data, 1).reshape((gol.info.width, gol.info.height)))
            self.axes[i//3, i%3].set_title(gol.header.frame_id)

        # Merge the GOL
        GOLout = copy.deepcopy(self.usersGOL[0])
        rawMap = np.full((GOLout.info.width*GOLout.info.height), -1, dtype=float)
        tic = time.process_time()

        # Average on know cells
        # for i in range(len(rawMap)):
        #     avg = 0.0
        #     cnt = 0
        #     for gol in self.usersGOL:
        #         if gol.data[i] != -1:
        #             cnt += 1.0
        #             avg += gol.data[i]
        #     if cnt != 0:
        #         rawMap[i] = avg / cnt

        # DST
        for i in range(len(rawMap)):
            combined_masses = None
            for gol in self.usersGOL:
                
                map_masses = self.bba1(gol.data[i], gol.header.frame_id)

                if combined_masses == None:
                    combined_masses = copy.deepcopy(map_masses)
                else:
                    combined_masses = combined_masses.sum(map_masses)

            rawMap[i] = combined_masses.get_mass({"O"}) * 100.0

        # Up Contrast
        # maxi = np.amax(rawMap)
        # rawMap = np.multiply(rawMap, 100.0/maxi)

        # Clip the map between values for ROS standard
        rawMap = np.maximum(rawMap, -1)
        rawMap = np.minimum(rawMap, 100)
        toc = time.process_time()
        rospy.loginfo("Merged {} GOL of {} in {}s.".format(len(self.usersGOL), data.header.frame_id, toc-tic))
        

        GOLout.data = rawMap.astype(dtype=np.int8).flatten().tolist()
        self.pub.publish(GOLout)

        # displaying
        mapimg = np.add(GOLout.data, 1).reshape((GOLout.info.width, GOLout.info.height))
        self.axes[1, 2].imshow(mapimg)
        self.axes[1, 2].set_title("Merged GOL")
        plt.pause(0.01)
        # GOLout.header.frame_id


    def bba1(self, cell, frame_id):
        O = 0
        F = 0
        OF = 0
        # if is from infrastructure 
        if frame_id.find('Infra') == -1:
            if cell == -1:
                OF = 1.0
            else: 
                F = 1.0 - (cell / 100.0)
                OF = (cell / 100.0)
        # if is from embeded sensor 
        else:
            if cell == -1:
                OF = 1.0
            else: 
                O = (cell / 100.0)
                OF = 1.0 - (cell / 100.0)
                    
        return DST([({"O"}, O), ({"F"}, F), ({"O", "F"}, OF)])


    def bba0(self, cell, frame_id):
        O = 0
        F = 0
        OF = 0
        # if is from infrastructure 
        if frame_id.find('Infra') == -1:
            if cell == -1:
                OF = 1.0
            else: 
                O = (cell / 100.0)
                OF = 1.0 - O
                
        # if is from embeded sensor 
        else:
            if cell == -1:
                OF = 1.0
            else:
                F = 1.0 - (cell / 100.0)
                OF = 1 - F
                    
        return DST([({"O"}, O), ({"F"}, F), ({"O", "F"}, OF)])

if __name__ == '__main__':
    proj_node = GOmerger()


