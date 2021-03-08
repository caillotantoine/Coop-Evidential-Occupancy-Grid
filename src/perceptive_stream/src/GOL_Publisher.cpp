#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Pose.h>
#include "perceptive_stream/BBox3D.h"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <boost/foreach.hpp>

#define DEBUG

#define MAP_WIDTH 16000
#define MAP_HEIGHT 16000

using namespace Eigen;
using namespace std;

vector<int8_t> raw_map;
nav_msgs::OccupancyGrid GOL;
geometry_msgs::Pose raw_mapOrigin;
ros::Publisher pub;

void bbox_rx_callback(const perceptive_stream::BBox3D::ConstPtr& bbox);

int main(int argc, char** argv)
{
    ros::init(argc, argv, "GOL_Publisher");
    ros::NodeHandle handler_TX;
    ros::NodeHandle handler_RX;
    pub = handler_TX.advertise<nav_msgs::OccupancyGrid>("gndTruth/GOL", 10);
    ros::Subscriber sub = handler_RX.subscribe("car/info", 10, bbox_rx_callback);
    ros::Rate loop_rate(10);

    ROS_INFO("Generating the map.");
    for (int64_t i = 0; i < (MAP_WIDTH * MAP_HEIGHT); i++)
    {
        raw_map.push_back(-1);
    }

    //Data size doesn't match width*height: width = 16000, height = 16000, data size = 256000001
    GOL.info.resolution = 1.0/8.0;
    GOL.info.width = MAP_WIDTH;
    GOL.info.height = MAP_HEIGHT;

    raw_mapOrigin.position.x = - MAP_WIDTH / 16;
    raw_mapOrigin.position.y = - MAP_HEIGHT / 16;
    raw_mapOrigin.position.z = 0;
    // We gonna ignore the roations for now, but it will be implemented later

    GOL.info.origin = raw_mapOrigin;
    GOL.data = raw_map;
    ROS_INFO("Map generation completed.");

    ros::spin();
    return 0;
}


void bbox_rx_callback(const perceptive_stream::BBox3D::ConstPtr& bbox)
{
    ROS_INFO("Received BBox : %s", bbox->header.frame_id.c_str());
    stringstream debug;

    double l = bbox->size.x;
    double w = bbox->size.y;
    double h = bbox->size.z;

    vector<Vector4d> bbox_corners;
    bbox_corners.push_back(Vector4d(l, -w, -h, 1.0));
    bbox_corners.push_back(Vector4d(l, w, -h, 1.0));
    bbox_corners.push_back(Vector4d(-l, w, -h, 1.0));
    bbox_corners.push_back(Vector4d(-l, -w, -h, 1.0));

    Matrix4d bbox_center(4, 4);
    Vector4d bbox_t(bbox->center.position.x,
                    bbox->center.position.y,
                    bbox->center.position.z, 
                    1.0);
    bbox_center.col(3) = bbox_t;

#ifdef DEBUG
    debug << "BBOX mat : \n" << bbox_center << endl;
    ROS_INFO_ONCE(debug.str().c_str());
#endif

    Vector4d vehicle_t(bbox->vehicle.position.x,
                       bbox->vehicle.position.y,
                       bbox->vehicle.position.z, 
                       1.0);
    
    Quaterniond vehicle_rot = Quaterniond(bbox->vehicle.orientation.w,
                                         bbox->vehicle.orientation.x,
                                         bbox->vehicle.orientation.y,
                                         bbox->vehicle.orientation.z);
    Matrix4d vehicle_pos;
    vehicle_pos.block<3, 3>(0, 0) = vehicle_rot.normalized().toRotationMatrix();
    vehicle_pos.col(3) = vehicle_t;

#ifdef DEBUG
    flush(debug);
    debug << "Vehicle mat : \n" << vehicle_pos << endl;
    ROS_INFO_ONCE(debug.str().c_str());
#endif

    vector<Vector4d> new_bbox_corners;
    
    BOOST_FOREACH(Vector4d pt, bbox_corners)
    {
        Vector4d out;
        // out = vehicle_pos * bbox_center * pt.transpose();
#ifdef DEBUG
        flush(debug);
        debug << "Point : \n" << vehicle_pos * bbox_center * pt << endl;
        ROS_INFO_ONCE(debug.str().c_str());
#endif
    }

    pub.publish(GOL);
}
