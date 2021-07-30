#!/usr/bin/env python3
# coding: utf-8

import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class GOmerger:

    def __init__(self):
        rospy.init_node("GO_merger", anonymous=True)
        rospy.Subscriber('projector/GOL', OccupancyGrid, self.callback)
        self.pub = rospy.Publisher('merger/GO', OccupancyGrid, queue_size=10)

        self.usersGOL = []

        rospy.spin()

    def callback (self, data: OccupancyGrid):
        # rospy.logwarn("GOL Merger =============================\n{}\n".format(data.header))
        GOLfound = False
        for i,gol in enumerate(self.usersGOL):
            if data.header.frame_id.split('.')[0] == gol.header.frame_id.split('.')[0]:
                # rospy.logerr("Found! replace!")
                self.usersGOL[i] = data
                GOLfound = True
                break
        if not GOLfound:
            self.usersGOL.append(data)

        # Merge the GOL
        GOLout = self.usersGOL[0]
        rawMap = np.full((GOLout.info.width*GOLout.info.height), -1, dtype=float)
        tic = time.process_time()
        for gol in self.usersGOL:
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


if __name__ == '__main__':
    proj_node = GOmerger()