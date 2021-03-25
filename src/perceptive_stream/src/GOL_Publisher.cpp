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

#include "GOL_proj.h"

#define DEBUG

#define MAP_WIDTH 800
#define MAP_HEIGHT 800

using namespace Eigen;
using namespace std;

vector<int8_t> raw_map(MAP_WIDTH * MAP_HEIGHT);
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
    fill(raw_map.begin(), raw_map.end(), -1);

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
    flush(debug);
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
    double x_min = 100, x_max = -100, y_min = 100, y_max = -100; //ROI limits TBD

    // 
    //      RASTERISATION
    //          On definie la zone dans laquel se trouve les points (via les min et les max de X et Y)
    //          à partir des points: on creer des equations de droites pour faire les conditions sur la présence des points
    //          on tire des points sur le centre de chaques cases et on regarde si on est dans la forme ou pas. 
    //          (on pourra tirer plusieurs points, au quatres coins par exemple, pour faire varier l'intensité)
    //
    
    BOOST_FOREACH(Vector4d pt, bbox_corners)
    {
        Vector4d out;
        out = vehicle_pos * bbox_center * pt;       // projette le point dans le repère monde

        // fenetre d'interêt pour ne pas faire toutes les cases
        if(out[0] < x_min)
            x_min = out[0];
        if(out[0] > x_max)
            x_max = out[0];
        if(out[1] < y_min)
            y_min = out[1];
        if(out[1] > y_max)
            y_max = out[1];

        new_bbox_corners.push_back(out);
#ifdef DEBUG
        flush(debug);
        debug << "Point : \n" << out << endl;
        ROS_INFO_ONCE(debug.str().c_str());
#endif
    }

    vector<line2D> object_boundaries;
    for(int i = 0; i < new_bbox_corners.size(); i++)
    {
        Vector4d A, B, C;
        line2D l;
        A = new_bbox_corners[i];
        B = new_bbox_corners[(i+1)%new_bbox_corners.size()];

        if(abs(A[0] - B[0]) <= 0.001) // check if vertical
        {
            l.vertical = true;
            l.a = A[0];
            l.b = 0;
            continue;
        }
        l.vertical = false;

        if(A[0] > B[0])
        {
            C = A;
            A = B;
            B = C;
        }

        l.a = (B[1] - A[1]) / (B[0] - A[0]);
        l.b = A[1] - (l.a * A[0]); //check if correct

        // create a set of lines
        object_boundaries.push_back(l);
    }

    // https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/

    pub.publish(GOL);
}
