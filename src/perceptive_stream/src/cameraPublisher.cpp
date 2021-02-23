#include "ros/ros.h"
#include "std_msgs/String.h"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>


#include <sstream>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Camera_Publisher");

    ros::NodeHandle n;

    ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
    ros::Rate loop_rate(20);

    int count = 0;

    while(ros::ok()) 
    {
        std_msgs::String msg;

        std::stringstream ss;
        ss << "Hello world " << count << " " << argv[1];
        msg.data = ss.str();

        ROS_INFO("%s", msg.data.c_str());

        chatter_pub.publish(msg);
        ros::spinOnce();

        loop_rate.sleep();
        count++;
    }

    return 0;
}