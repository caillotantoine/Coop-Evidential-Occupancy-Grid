#include <stdio.h>
#include <stdlib.h>

#include <iostream>

// #include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

int main(int argc, char **argv)
{
    Mat img = imread("/home/caillot/Documents/Dataset/CARLA_Dataset_A/Infra/cameraRGB/000051.png", IMREAD_COLOR);
    imshow("Display image", img);
    waitKey(0);
    printf("test\n");
    return 0;
}