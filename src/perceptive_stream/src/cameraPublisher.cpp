#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>


#include <sstream>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Camera_Publisher");
    ros::NodeHandle n;

    image_transport::ImageTransport img_trans(n);
    image_transport::Publisher pub = img_trans.advertise("camera/image", 5);
    ros::Publisher camInfo_pub = n.advertise<sensor_msgs::CameraInfo>("camrea/info", 5);

    ros::Rate loop_rate(20);


    cv::Mat img;
    sensor_msgs::ImagePtr img_msg;

    img = cv::imread(argv[1]);


    int count = 0;

    while(ros::ok()) 
    {
        if(!img.empty()){
            img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
            pub.publish(img_msg);
            ROS_INFO("Showing %s", argv[1]);
        }

        

        ros::spinOnce();
        loop_rate.sleep();
        count++;
    }

    return 0;
}