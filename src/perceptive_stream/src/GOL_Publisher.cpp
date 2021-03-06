#include <ros/ros.h>
#include <sstream>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Pose.h>

#define MAP_WIDTH 16000
#define MAP_HEIGHT 16000

int main(int argc, char** argv)
{
    nav_msgs::OccupancyGrid GOL;
    geometry_msgs::Pose mapOrigin;
    ros::NodeHandle handler;

    int8_t map[MAP_WIDTH * MAP_HEIGHT] = {0}; // Must be a vector: TO FIX


    ros::init(argc, argv, "GOL_Publisher");
    ros::Publisher pub = handler.advertise<nav_msgs::OccupancyGrid>("gndTruth/GOL", 10);

    

    GOL.info.resolution = 1/8;
    GOL.info.width = MAP_WIDTH;
    GOL.info.height = MAP_HEIGHT;

    mapOrigin.position.x = 0;
    mapOrigin.position.y = 0;
    mapOrigin.position.z = 0;
    // We gonna ignore the roations for now, but it will be implemented later

    GOL.info.origin = mapOrigin;
    GOL.data = map;


    return 0;
}