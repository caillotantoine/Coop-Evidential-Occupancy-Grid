import rospy
import numpy as np
import cv2 as cv
from threading import Lock
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pyquaternion import Quaternion

from perceptive_stream.msg import Img, BBox3D

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


class BBoxProj:

    # pub : the ros publisher
    # Bbox_queue : the Queue of the previous bbox
    # cv_bridge : bridge openCV <-> ROS

    def __init__(self):
        # init ROS 
        rospy.init_node("bbox_reproj", anonymous=True)
        queue_size = rospy.get_param('~queue_size')

        # create a publisher
        self.pub = rospy.Publisher('projector/img', Image, queue_size=10)
        self.pub_mutex = Lock()

        # Creat an OpenCV bridge
        self.cv_bridge = CvBridge()

        # Create a queue for the bounding box. Used to find a matching Bbox with an image
        self.Bbox_queue = QueueROSmsgs(queue_size, debug=True)

        rospy.Subscriber('camera/image', Img, self.callback_img)
        rospy.Subscriber('car/info', BBox3D, self.callback_bbox)

        rospy.spin()

    def callback_img(self, data):
        # Display and record the frame ID number (used to synchronize)
        img_id = data.header.frame_id.split('.')[1]
        rospy.loginfo("Received img : %s", img_id)

        # There can be several bounding box (one per vehicles)
        listOfBBox = []
        bboxFound = True
        while(bboxFound):
            bboxIdx = self.Bbox_queue.searchFirst(".%s"%img_id) # Looking for corresponding bounding box with this image
            if bboxIdx == -1:
                bboxFound = False 
                break # there is no bounding box anymore, let's get out of the loop
            listOfBBox.append(self.Bbox_queue.pop(bboxIdx)) # remove the corresponding bounding box from the que and add it to the local list

        img_msg = Image()

        if len(listOfBBox) > 0:
            # If there are some bounding box, draw them on the image
            img_msg = self.drawBoxes(data, listOfBBox)
        else:
            # if there is no bounding box, just pipe the image
            img_msg = data.image

        img_msg.header = data.header

        self.pub_mutex.acquire()        # Is it mendatory?
        try:
            self.pub.publish(img_msg) # publish
        finally:
            self.pub_mutex.release()
        

    def callback_bbox(self, data):
        self.Bbox_queue.add(data) # just add the read bounding box to the queue

    def drawBoxes(self, image, bboxes):
        # take as input the image and the list of bounding box to draw
        # apply the draw to the image

        # convert the Image message to opoenCV format
        cv_img = self.cv_bridge.imgmsg_to_cv2(image.image, "bgr8")

        # information about the image an drawing parameters
        h, w, depth = cv_img.shape
        color = (0, 255, 0)
        thickness = 2


        for bbox in bboxes:
            #for each bounding box

            bbox_vert_world = self.placeBBoxInWorld(bbox) # place the 8 corner of the box in the world
            bbox_vert_cam = self.compute1bbox(image, bbox_vert_world)   # project the points in the camera plane

            points = [(int(x[0]), int(x[1])) for x in bbox_vert_cam] # we need tuple for the points

            rospy.loginfo("vertex :\n%s"%points)  

            # There can be no points if the bounding box is behind the camera. 
            if len(points) > 0:
                # draw each edges
                for idx in range(len(points)-1):
                    cv_img = cv.line(cv_img, points[idx], points[idx+1], color, thickness)
                cv_img = cv.line(cv_img, points[0], points[5], color, thickness)
                cv_img = cv.line(cv_img, points[1], points[6], color, thickness)
                cv_img = cv.line(cv_img, points[2], points[7], color, thickness)
                cv_img = cv.line(cv_img, points[4], points[7], color, thickness)
                cv_img = cv.line(cv_img, points[0], points[3], color, thickness)

        return self.cv_bridge.cv2_to_imgmsg(cv_img, "bgr8") # convert the openCV image to a ROS Image message

    
    def compute1bbox(self, image, bbox):
        # get intrinsec parameters (K) matrix of the camera.
        K = np.array(image.info.K)
        K = np.reshape(K, (3, 3))

        # Get the position of the camera
        T_WCw = pose2Tmat(image.pose) # cam -> world
        T_CwW = np.linalg.inv(T_WCw) # world -> cam

        T_CCw = getTCCw() # world system to camera system 

        T_CW = np.matmul(T_CCw, T_CwW) # world -> camera

        vertex = []
        for point in bbox:
            in_cam_space = np.matmul(T_CW, point) # place the point in camera space
            in_cam_space_cropped = in_cam_space[:3] # crop to get a 1x3 vector
            in_px_raw = np.matmul(K, in_cam_space_cropped) # project the point in the camera plane
            in_px_norm = in_px_raw / in_px_raw[2] # normalize the projection
            if in_px_raw[2] < 0.0: # if the point is behind the camera (z < 0), return an empty set of points
                return []
            vertex.append(in_px_norm) 
        return vertex

    def placeBBoxInWorld(self, bbox):

        # bounding box size
        l = bbox.size.x
        w = bbox.size.y
        h = bbox.size.z

        # Bounding box vertex (following CARLA description)
        vertex = []
        vertex.append(np.array([l, -w, -h, 1.0]))
        vertex.append(np.array([l, w, -h, 1.0]))
        vertex.append(np.array([-l, w, -h, 1.0]))
        vertex.append(np.array([-l, -w, -h, 1.0]))
        vertex.append(np.array([-l, -w, h, 1.0]))
        vertex.append(np.array([l, -w, h, 1.0]))
        vertex.append(np.array([l, w, h, 1.0]))
        vertex.append(np.array([-l, w, h, 1.0]))

        # Get Tmat to move the points around the car model at 0, 0, 0
        T_VB = np.identity(4)
        T_VB[0][3] = bbox.center.position.x
        T_VB[1][3] = bbox.center.position.y
        T_VB[2][3] = bbox.center.position.z

        # Apply to the vertex the transformation. The center of the Bounding Box and the vehicle location are different.
        for idx, v in enumerate(vertex):
            vertex[idx] = np.matmul(T_VB, v)
        
        # get Tmat vehicle -> world
        T_WV = pose2Tmat(bbox.vehicle)

        # Apply to the vertex the transformation. place every vertex in the world
        for idx, v in enumerate(vertex):
            vertex[idx] = np.matmul(T_WV, v)

        return vertex

def pose2Tmat(pose):
    # get the position of the camera
    R_Q = Quaternion(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z) 
    T_mat = R_Q.transformation_matrix   # Transfromation only fitted with rotation elements
    T_mat[0][3] = pose.position.x   # Adding the translation elements
    T_mat[1][3] = pose.position.y
    T_mat[2][3] = pose.position.z
    return T_mat

def getTCCw():
    return np.array([[0.0,   1.0,   0.0,   0.0,], [0.0,   0.0,  -1.0,   0.0,], [1.0,   0.0,   0.0,   0.0,], [0.0,   0.0,   0.0,   1.0,]])
    # matrix to change from world space to camera space 



if __name__ == '__main__':
    proj_node = BBoxProj()