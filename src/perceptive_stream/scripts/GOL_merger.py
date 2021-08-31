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
import multiprocessing as mp

def bba1(cell, frame_id):
    O = 0
    F = 0
    OF = 0
    
    if cell == -1:
        OF = 1.0
    else:
        # if is from infrastructure  == -1
        if frame_id.find('Infra') == -1:
            F = 1.0 - (cell / 100.0)
            OF = (cell / 100.0)
        # if is from embeded sensor 
        else:
            O = (cell / 100.0)
            OF = 1.0 - (cell / 100.0)
                    
    return DST([({"O"}, O), ({"F"}, F), ({"O", "F"}, OF)])


def bba0(cell, frame_id):
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

def avg0(cells):
    avg = 0.0
    cnt = 0
    for (cell, _) in cells:
        if cell != -1:
            cnt += 1
            avg += cell
    if cnt != 0:
        return max(0, min(avg / cnt, 100))
    else:
        return -1

def avg1(cells):
    c = None
    cnt = 0
    for (cell, _) in cells:
        if cell != -1:
            cnt += 1
            p = cell / 100.0
        else:
            p = 0.5
        if c != None:
            c *= p
        else:
            c = p
    if cnt > 0:
        return max(0, min(c * 100.0, 100))
    else:
        return -1

def avg2(cells):
    c = None
    cnt = 0
    for (cell, _) in cells:
        if cell != -1:
            cnt += 1
            p = cell / 100.0
            if c != None:
                c *= p
            else:
                c = p
    if cnt > 0:
        return max(0, min(c * 100.0, 100))
    else:
        return -1


def dst1(cells):
    combined_masses = None
    for (cell, frame_id) in cells:
        map_masses = bba1(cell, frame_id)

        if combined_masses == None:
            combined_masses = copy.deepcopy(map_masses)
        else:
            combined_masses = combined_masses.sum(map_masses)

    return min(combined_masses.get_mass({"O"}) * 100.0, 100)

def dst2(cells):
    combined_masses = None
    for (cell, frame_id) in cells:
        map_masses = bba1(cell, frame_id)

        if combined_masses == None:
            combined_masses = copy.deepcopy(map_masses)
        else:
            combined_masses = combined_masses.sum(map_masses)

    O = combined_masses.get_mass({"O"})
    F = combined_masses.get_mass({"F"})
    OF = combined_masses.get_mass({"O", "F"})

    if OF > O and OF > F:
        return -1
    if F > O and F > OF:
        return 0
    return 100

class GOmerger:

    def __init__(self):
        rospy.init_node("GO_merger", anonymous=True)
        rospy.Subscriber('projector/GOL', OccupancyGrid, self.callback)
        self.pub = rospy.Publisher('merger/GO', OccupancyGrid, queue_size=10)
        self.usersGOL = []

        self.fig = None
        self.last_id = -1


        rospy.spin()

    def callback (self, data: OccupancyGrid):
        # rospy.logwarn("GOL Merger =============================\n{}\n".format(data.header))
        GOLfound = False
        maps_instock = []
        for m in self.usersGOL:
            maps_instock.append(m.header.frame_id)
        # rospy.logerr("New map received : {}\nOld ones : {}".format(data.header.frame_id, maps_instock))
        
        if int(data.header.frame_id.split('.')[1]) > self.last_id and self.last_id != -1:
            rospy.loginfo("Start merging maps. ID: {}".format(self.last_id))
            self.last_id = int(data.header.frame_id.split('.')[1])

            # if self.fig is None:
            #     self.fig, self.axes = plt.subplots(2, 3)
            #     self.fig.suptitle("Occupancy grids")

            # for i,gol in enumerate(self.usersGOL):
            #     # rospy.logerr("Show : {} on [{}, {}] (i = {})".format(gol.header.frame_id, i//3, i%3, i))
            #     self.axes[i//3, i%3].imshow(np.add(gol.data, 1).reshape((gol.info.width, gol.info.height)))
            #     self.axes[i//3, i%3].set_title(gol.header.frame_id)

            # Merge the GOL
            GOLout = copy.deepcopy(self.usersGOL[0])
            rawMap = np.full((GOLout.info.width*GOLout.info.height), -1, dtype=float)
            pool = mp.Pool(mp.cpu_count())
            tic = time.process_time()

            gols = []
            for i in range(len(rawMap)):
                gols.append([(x.data[i], x.header.frame_id) for x in self.usersGOL])



            rawMap = pool.map(dst2, gols)

            # Clip the map between values for ROS standard
            toc = time.process_time()
            rospy.loginfo("Merged {} GOL of {} in {}s.".format(len(self.usersGOL), self.last_id, toc-tic))
            

            GOLout.data = rawMap
            self.pub.publish(GOLout)

            # displaying
            mapimg = np.add(GOLout.data, 1).reshape((GOLout.info.width, GOLout.info.height))
            # self.axes[1, 2].imshow(mapimg)
            # self.axes[1, 2].set_title("Merged GOL {}".format(GOLout.header.frame_id.split('.')[1]))
            # plt.pause(0.01)

            plt.imsave("/home/caillot/Bureau/GOL_images/gol{}.png".format(GOLout.header.frame_id.split('.')[1]), mapimg, vmin=-1, vmax=100, cmap='Greys')
            # GOLout.header.frame_id

        self.last_id = int(data.header.frame_id.split('.')[1])
        for i,gol in enumerate(self.usersGOL):
            if data.header.frame_id.split('.')[0] == gol.header.frame_id.split('.')[0]:
                # rospy.logerr("Found! replace!")
                self.usersGOL[i] = data
                GOLfound = True
                break
        if not GOLfound:
            self.usersGOL.append(data)


if __name__ == '__main__':
    proj_node = GOmerger()


