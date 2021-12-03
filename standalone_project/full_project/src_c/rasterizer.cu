#include <stdio.h>
#include <stdlib.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define MAPSIZE (120.0)
#define GRIDSIZE (120*5)

void bonjour_cpp();
void projector_cpp(int len, int* labels, float* fp_vec, unsigned char* map);

extern "C" {
    void bonjour(){bonjour_cpp();}
}

using namespace cv;

int main(int argc, char **argv)
{
    float fp_vec[1][4][2] = {{{7.73851759, -49.70380452}, {4.39891667, -49.70377771}, {3.45951359, -38.3454241}, {6.0859267,  -38.34544071}}};
    int label[1] = {1};

    Mat map = Mat::zeros(GRIDSIZE, GRIDSIZE, CV_8U);

    int i, j;
    for (i = 0; i<4; i++)
    {
        printf("<%3.2f; %3.2f>\n", fp_vec[0][i][0], fp_vec[0][i][1]);
    }
    printf("test nvcc!!!!!\n");
    ellipse(map, Point(GRIDSIZE/4, GRIDSIZE/4), Size(GRIDSIZE/4, GRIDSIZE/4), 0, 0, 360, 0xff, 2, 8);
    projector_cpp(1, label, (float*) fp_vec, map.ptr());

    imshow("Test map display", map);
    waitKey(0);
    return 0;
}

void bonjour_cpp()
{
    printf("Bonjour!!!\n");
}

void projector_cpp(int len, int* labels, float* fp_vec, unsigned char* map)
{
    int i, j;
    Mat n_map = Mat(GRIDSIZE, GRIDSIZE, CV_8U, map);
    Point fp[1][4];
    float fx, fy;
    int x, y;
    float* addr_x, addr_y;

    ellipse(n_map, Point(GRIDSIZE/2, GRIDSIZE/2), Size(GRIDSIZE/4, GRIDSIZE/4), 0, 0, 360, 0xff, 2, 8);
    
    // printf("%3.2f\n", *(fp_vec + 3));
    for(i=0; i<len; i++)
    {
        for(j=0; j<4; j++)
        {
            printf("Label %d: ", labels[i]);
            addr_x = (fp_vec + 8*i + 2*j);
            addr_y = (fp_vec + 8*i + 2*j + 1);
            fx = *addr_x;
            fy = *addr_y;
            x = 0;
            y = 0;
            printf("<%3.2f; %3.2f> \t <%d; %d> \t(%d, %d) \n", fx, fy, x, y, addr_x, addr_y);
            
        }
    }

    memcpy(map, n_map.ptr(), GRIDSIZE*GRIDSIZE);
    delete n_map;
}