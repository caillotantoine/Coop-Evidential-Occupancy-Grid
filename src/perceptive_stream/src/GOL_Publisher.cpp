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
#include <algorithm>

#include "opencv4/opencv2/opencv.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"


#define DEBUG

#define MAP_WIDTH 800
#define MAP_HEIGHT 800

using namespace Eigen;
using namespace std;
using namespace cv;

vector<int8_t> raw_map(MAP_WIDTH * MAP_HEIGHT);
nav_msgs::OccupancyGrid GOL;
geometry_msgs::Pose raw_mapOrigin;
ros::Publisher pub;

Mat out_img;

vector<string> vehicle_history;

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

    // namedWindow("GOLViz", 1);
    ros::spin();
    // waitKey(0);
    return 0;
}


void bbox_rx_callback(const perceptive_stream::BBox3D::ConstPtr& bbox)
{
    ROS_INFO("Received BBox : %s", bbox->header.frame_id.c_str());
    stringstream debug;

    string v_id = bbox->header.frame_id.substr(0, bbox->header.frame_id.find("."));
    if(find(vehicle_history.begin(), vehicle_history.end(), v_id) != vehicle_history.end())
    {
        ROS_INFO(v_id.c_str());
        vehicle_history.clear();
        out_img = Mat::zeros(MAP_HEIGHT, MAP_WIDTH, CV_8UC1);
    }
    vehicle_history.push_back(v_id);

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

    vector<Vector2i> new_bbox_corners;

    cv::Point pts2draw[1][30];
    int cnt = 0;
    BOOST_FOREACH(Vector4d pt, bbox_corners)
    {
        Vector4d out;
        // out = vehicle_pos * bbox_center * pt;       // projette le point dans le repère monde
        out = vehicle_pos * pt;

        // On passe de metrique au systeme de la grille d'occupation.
        Vector4d out_rGO = (1.0 / GOL.info.resolution) * out;
        Vector2i out_rGO_i;
        out_rGO_i << (int) -out_rGO[0] + MAP_WIDTH/2, (int) out_rGO[1] + MAP_HEIGHT / 2;

        // new_bbox_corners.push_back(out_rGO_i);
        if (cnt < 30)
        {
            pts2draw[0][cnt] = cv::Point(out_rGO_i[0], out_rGO_i[1]);
            cnt++;
        }
#ifdef DEBUG
        flush(debug);
        debug << "Point : \n" << out << endl << out_rGO_i << endl;
        ROS_INFO_ONCE(debug.str().c_str());
#endif
    }

    const cv::Point* ptset[1] = {pts2draw[0]};

    int npts[] = {cnt};
    cv::fillPoly(out_img, ptset, npts, 1, cv::Scalar(100), cv::LINE_8);
    // stringstream filename;
    // filename << "/home/caillot/testIMG/carGOL_" << bbox->header.frame_id << ".png";
    // cv::imwrite(filename.str().c_str(), out_img);
    // cv::imshow("GOLViz", out_img);

    if(out_img.isContinuous())
    {
        raw_map.assign(out_img.data, out_img.data + out_img.total());
    }

    GOL.data = raw_map;

    pub.publish(GOL);
}
