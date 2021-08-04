#!/usr/bin/env python3
# coding: utf-8

import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
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

        for i,gol in enumerate(self.usersGOL):
            # pass
            gol_data = np.array(gol.data)    
            rawMap = np.add(rawMap, gol_data)
        rawMap = np.divide(rawMap, len(self.usersGOL))
        rawMap = np.maximum(rawMap, -1)
        rawMap = np.minimum(rawMap, 100)
        # rawMap = np.divide(rawMap, len(self.usersGOL))
        toc = time.process_time()
        rospy.loginfo("Merged {} GOL of {} in {}s.".format(len(self.usersGOL), data.header.frame_id, toc-tic))
        

        

        GOLout.data = rawMap.astype(dtype=np.int8).flatten().tolist()
        self.pub.publish(GOLout)

        mapimg = np.add(GOLout.data, 1).reshape((GOLout.info.width, GOLout.info.height))
        self.axes[1, 2].imshow(mapimg)
        self.axes[1, 2].set_title("Merged GOL")
        

        plt.pause(0.01)
        
        

        # if self.mapFig is None:
        #     self.mapFig = plt.figure(1)
        #     self.ax = self.mapFig.add_subplot( 111 )
        #     self.ax.set_title(data.header.frame_id)
        #     self.im = self.ax.imshow(mapimg) # Blank starting image
        #     self.mapFig.show()
        #     self.im.axes.figure.canvas.draw()
        # else:
        #     self.ax.set_title(data.header.frame_id)
        #     self.im.set_data(mapimg)
        #     self.im.axes.figure.canvas.draw()

if __name__ == '__main__':
    proj_node = GOmerger()