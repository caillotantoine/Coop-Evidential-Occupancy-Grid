#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Pose.h>
#include "perceptive_stream/BBox3D.h"

#include <eigen3/Eigen/Dense>

#include <sstream>
#include <vector>

#define MAP_WIDTH 16000
#define MAP_HEIGHT 16000

using namespace Eigen;
using namespace std;

class BBox2GOL {
    private:
        ros::Publisher pub;
        ros::Subscriber sub;
        nav_msgs::OccupancyGrid GOL;
        geometry_msgs::Pose mapOrigin;
        

    public:
        BBox2GOL(int argc, char **argv, const char *node_name) 
        {
            ros::init(argc, argv, node_name);
            ros::NodeHandle handler_TX;
            ros::NodeHandle handler_RX;
            this.pub = handler_TX.advertise<nav_msgs::OccupancyGrid>("gndTruth/GOL", 10);
            this.sub = handler_RX.subscribe("car/info", 10, this.bbox_rx_callback);

            for (int64_t i = 0; i < (MAP_WIDTH * MAP_HEIGHT); i++)
            {
                this.map.push_back(-1);
            }

            this.GOL.info.resolution = 1.0/8.0;
            this.GOL.info.width = MAP_WIDTH;
            this.GOL.info.height = MAP_HEIGHT;

            this.mapOrigin.position.x = - MAP_WIDTH / 16;
            this.mapOrigin.position.y = - MAP_HEIGHT / 16;
            this.mapOrigin.position.z = 0;

            this.GOL.info.origin = mapOrigin;
            this.GOL.data = map;
        }

        void bbox_rx_callback(const perceptive_stream::BBox3D::ConstPtr &msg)
        {
            //none
        }
};



int main(int argc, char** argv)
{
    nav_msgs::OccupancyGrid GOL;
    geometry_msgs::Pose mapOrigin;

    vector<int8_t> map; // Must be a vector: TO FIX


    ros::init(argc, argv, "GOL_Publisher");
    ros::NodeHandle handler_TX;
    ros::NodeHandle handler_RX;
    ros::Publisher pub = handler_TX.advertise<nav_msgs::OccupancyGrid>("gndTruth/GOL", 10);
    ros::Subscriber sub = handler_RX.subscribe("car/info", 10, bbox_rx_callback);
    ros::Rate loop_rate(10);

    for (int64_t i = 0; i < (MAP_WIDTH * MAP_HEIGHT); i++)
    {
        map.push_back(-1);
    }

    //Data size doesn't match width*height: width = 16000, height = 16000, data size = 256000001

    GOL.info.resolution = 1.0/8.0;
    GOL.info.width = MAP_WIDTH;
    GOL.info.height = MAP_HEIGHT;

    mapOrigin.position.x = - MAP_WIDTH / 16;
    mapOrigin.position.y = - MAP_HEIGHT / 16;
    mapOrigin.position.z = 0;
    // We gonna ignore the roations for now, but it will be implemented later

    GOL.info.origin = mapOrigin;
    GOL.data = map;

    while(ros::ok())
    {
        pub.publish(GOL);
        ros::spinOnce();
        ROS_INFO_ONCE("Published first map\n");

        loop_rate.sleep();
    }


    return 0;
}
