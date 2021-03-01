import rospy
import numpy as np
import cv2 as cv
from threading import Lock
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from perceptive_stream.msg import Img, BBox3D

received_bbox = []
mutex_bboxes = Lock()

mutex_pub = Lock()

def projector(img_msg, bbox_msg):
    # TODO : It may be necessary to filter the stream we want to display. With parameters? 

    bridge = CvBridge() # TODO : Optimese this. Creating a class?

    # converting the image from ROS to OpenCV
    cv_img = bridge.imgmsg_to_cv2(img_msg.image, "bgr8")

    # OpenCV stuff : drawing a diagonal line (testing purpose)
    img_h, img_w, img_c = cv_img.shape
    start_point = (0, 0)
    end_point = (img_w, img_h)
    color = (0, 255, 0)

    cv_img = cv.line(cv_img, start_point, end_point, color, 9)

    # Convert OpenCV image to ROS image
    ros_image = Image()
    ros_img = bridge.cv2_to_imgmsg(cv_img, "bgr8")
    ros_img.header = img_msg.header # Not sure it is necessary. But why not

    # Publishing the image. There can be a race condition here is several callbacks are called in same time. 
    global pub
    pub.publish(ros_img)



def callback_img(data):
    # Display and record the frame ID number (used to synchronize)
    img_id = int(data.header.frame_id.split('.')[1])
    rospy.loginfo("Received img : %d", img_id)

    # Look for a bbox with a frame id number corresponding to the image's frame ID number
    mutex_bboxes.acquire()
    try:
        global received_bbox
        found = False
        # TODO : There can be several corresponding BBox, therefore it is needed to manage a list instead
        for i, bbox in enumerate(received_bbox):
            if int(bbox.header.frame_id.split('.')[1]) == img_id: 
                rospy.loginfo("found at idx %d"%i)
                projector(data, bbox)
                found = True
        if not found:
            rospy.loginfo("No corresponding bounding box found.")
    finally:
        mutex_bboxes.release()


def callback_bbox(data):
    # Display the frame ID number (used to synchronize)
    bbox_id = int(data.header.frame_id.split('.')[1])
    rospy.loginfo("Received bbox : %d", bbox_id)

    # Add the BBox to the queue
    mutex_bboxes.acquire() # Avoid the race condition between the callbacks
    try: 
        global received_bbox
        received_bbox.append(data)
        if(len(received_bbox) > 100): # if the queue has to much elements, remove the oldest one
            trash = received_bbox.pop(0)
            rospy.loginfo("Trashing %s", trash.header.frame_id)
    finally:
        mutex_bboxes.release()



def receiver_talker():
    # Init ROS
    rospy.init_node("bbox_reproj", anonymous=True)

    # Create a global publisher to output the camera stream annotated
    global pub
    pub = rospy.Publisher('projector/img', Image, queue_size=10)

    # Create subscribers for the camera and for the bounding boxes
    rospy.Subscriber('camera/image', Img, callback_img)
    rospy.Subscriber('car/info', BBox3D, callback_bbox)
        

    rospy.spin()


if __name__ == '__main__':
    receiver_talker()