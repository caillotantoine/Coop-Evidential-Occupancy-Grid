import sys
from os import path
import rospy
import json
from perceptive_stream.msg import BBox3D
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Vector3
from pyquaternion import Quaternion
import numpy as np

def talker():
    bbox_msg = BBox3D()
    cnt = 125

    rospy.init_node('carGNDtruth_Publisher', anonymous=True)
    path_to_json = rospy.get_param('~json_path')
    starting_point = rospy.get_param('starting')
    vehicle_ID = rospy.get_param('~vehicle_ID')

    rospy.loginfo("Path to json : %s"%path_to_json)
    rospy.loginfo("Starting point : %d"%starting_point)
    rospy.loginfo("Vehicle ID : %s"%vehicle_ID)


    pub = rospy.Publisher('car/info', BBox3D, queue_size=10)
    rate = rospy.Rate(20)


    rospy.loginfo('Start streaming')
    while not rospy.is_shutdown():
        jpath = path_to_json % (starting_point+cnt)
        rospy.loginfo("Trying : %s"%jpath)
        if path.exists(jpath):
            with open(jpath) as f:
                data = json.load(f)
            
            center = Pose()
            center.position.x = data['vehicle']['BoundingBox']['loc']['x']
            center.position.y = data['vehicle']['BoundingBox']['loc']['y']
            center.position.z = data['vehicle']['BoundingBox']['loc']['z']

            size = Vector3()
            size.x = data['vehicle']['BoundingBox']['extent']['x']
            size.y = data['vehicle']['BoundingBox']['extent']['y']
            size.z = data['vehicle']['BoundingBox']['extent']['z']

            vehicle = Pose()
            vehicleT = np.array(data['vehicle']['T_Mat'])
            vehicle_t = vehicleT[:3, 3:4]
            vehicle.position.x = vehicle_t.flatten()[0]
            vehicle.position.y = vehicle_t.flatten()[1]
            vehicle.position.z = vehicle_t.flatten()[2]

            vehicle_r = vehicleT[:3, :3]
            # In worlds coordinates, only yaw axis is used (otherwise this is called "an accident")
            theta = -np.arcsin(vehicle_r[2, 0])
            theta = np.pi - theta if abs(np.cos(theta) - 0.0) < 0.01 else theta
            yaw = np.arctan2(vehicle_r[1,0]/np.cos(theta), vehicle_r[0,0]/np.cos(theta))
            rospy.loginfo("Yaw : %f°" % np.degrees(yaw))
            vehicle_R = Quaternion(axis=[0.0, 0.0, 1.0], radians=yaw)
            vehicle.orientation.w = vehicle_R.w
            vehicle.orientation.x = vehicle_R.x
            vehicle.orientation.y = vehicle_R.y
            vehicle.orientation.z = vehicle_R.z

            bbox_msg.vehicle = vehicle
            bbox_msg.center = center
            bbox_msg.size = size
            bbox_msg.header.stamp = rospy.get_rostime()

            pub.publish(bbox_msg)
            cnt = cnt + 1
            rate.sleep()
        else:
            rospy.loginfo("Failed to read.")
            cnt = 0


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
