#!/usr/bin/env python3
# coding: utf-8

from threading import Lock
import rospy

class QueueROSmsgs:

    # size : size of the queue
    # queue : the list of data
    # mutex : protection in case of multiple call
    # debug : do we display the logs?


    def __init__(self, size, debug=False):
        self.size = size
        self.queue = []
        self.mutex = Lock()
        self.debug = debug

    def add(self, e):
        self.mutex.acquire()
        try:
            self.queue.append(e) # and an element at the top of the list
            if len(self.queue) > self.size: # if the list is too big:
                deleted_data = self.queue.pop(0) # delete the oldest element
                if self.debug:
                    rospy.loginfo("Deleted %s" % deleted_data.header.frame_id)
                
        finally:
            self.mutex.release()

    def searchFirst(self, frame_id):
        # Search the first element with a frame ID containing the string frame_id 
        # return the index in the list if found
        # return -1 otherwise
        to_ret = -1
        self.mutex.acquire()
        try:
            for idx, data in enumerate(self.queue):
                if data.header.frame_id.find(frame_id) > -1:
                    to_ret=idx
                    break
        finally:
            self.mutex.release()
        return to_ret


    def popAllPerv(self, queue_idx):
        # return the element at the given index and trash it from the list
        # trash every older element 
        self.mutex.acquire()
        try:
            to_ret = self.queue.pop(queue_idx)
            self.queue.pop([x for x in range(queue_idx)])
        finally:
            self.mutex.release()
        return to_ret

    def pop(self, idx=0):
        # return the element at the given index and trash it from the list
        self.mutex.acquire()
        try:
            to_ret = self.queue.pop(idx)
        finally:
            self.mutex.release()
        return to_ret