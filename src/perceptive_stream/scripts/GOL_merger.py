#!/usr/bin/env python3
# coding: utf-8

import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
from utils.DST import DST
import copy
import multiprocessing as mp
import signal

def timeout_handler(signum, frame):
    rospy.logerr("{} has timed out.".format(rospy.get_name()))
    raise Exception("{} has timed out.".format(rospy.get_name()))

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
        return int(max(0, min(avg / cnt, 100)))
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
        return int(max(0, min(c * 100.0, 100)))
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
        return int(max(0, min(c * 100.0, 100)))
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

    return int(min(combined_masses.get_mass({"O"}) * 100.0, 100))

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
        self.out_map_path = rospy.get_param('out_map_path')
        self.alg = rospy.get_param('~algo')
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
            signal.alarm(30)
            # Merge the GOL
            tic = time.process_time()
            GOLout = copy.deepcopy(self.usersGOL[0])
            rawMap = np.full((GOLout.info.width*GOLout.info.height), -1, dtype=float)
            pool = mp.Pool(mp.cpu_count()-2)
            

            gols = []
            for i in range(len(rawMap)):
                gols.append([(x.data[i], x.header.frame_id) for x in self.usersGOL])


            if self.alg == "avg1":
                rawMap = pool.map(avg1, gols)
            elif self.alg == "avg2":
                rawMap = pool.map(avg2, gols)
            elif self.alg == "dst1":
                rawMap = pool.map(dst1, gols)
            elif self.alg == "dst2":
                rawMap = pool.map(dst2, gols)
            else:
                rospy.logerr("{} is not a known algorithm".format(self.alg))
            pool.close()

            # Clip the map between values for ROS standard
            
            

            GOLout.data = rawMap
            self.pub.publish(GOLout)

            # displaying
            mapimg = np.add(rawMap, 1).reshape((GOLout.info.width, GOLout.info.height))
            # self.axes[1, 2].imshow(mapimg)
            # self.axes[1, 2].set_title("Merged GOL {}".format(GOLout.header.frame_id.split('.')[1]))
            # plt.pause(0.01)

            plt.imsave("{}/gol_{}.png".format(self.out_map_path,self.last_id), mapimg, vmin=-1, vmax=100, cmap='Greys')
            # GOLout.header.frame_id

            toc = time.process_time()
            rospy.loginfo("Merged {} GOL of {} in {}s.".format(len(self.usersGOL), self.last_id, toc-tic))

            if int(GOLout.header.frame_id.split('.')[1]) > 490:
                raise Exception("Done.")

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
    signal.signal(signal.SIGALRM, timeout_handler)
    proj_node = GOmerger()


