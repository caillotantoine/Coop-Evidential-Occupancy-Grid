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
    float fp_vec[2][4][2] = {{{7.73851759, -49.70380452}, {4.39891667, -49.70377771}, {3.45951359, -38.3454241}, {6.0859267,  -38.34544071}},
                             {{2.70641359, -48.17095769}, {0.8450105,  -48.17094317}, {0.72571429, -40.87830044}, {2.3243211,  -40.87831116}}};


    int label[2] = {1, 2};

    Mat map = Mat::zeros(GRIDSIZE, GRIDSIZE, CV_8UC4);

    int i, j;
    for (i = 0; i<4; i++)
    {
        printf("<%3.2f; %3.2f>\n", fp_vec[0][i][0], fp_vec[0][i][1]);
    }
    for (i = 0; i<4; i++)
    {
        printf("<%3.2f; %3.2f>\n", fp_vec[1][i][0], fp_vec[1][i][1]);
    }
    printf("test nvcc!!!!!\n");
    // ellipse(map, Point(GRIDSIZE/4, GRIDSIZE/4), Size(GRIDSIZE/4, GRIDSIZE/4), 0, 0, 360, 0xff, 2, 8);
    projector_cpp(2, label, (float*) fp_vec, map.ptr());

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
    Mat n_map = Mat(GRIDSIZE, GRIDSIZE, CV_8UC4, map);
    Point fp[1][4];
    float fx, fy;
    int x, y;
    float *addr_x, *addr_y;
    float stepgrid = ((float) MAPSIZE) / GRIDSIZE;

    // ellipse(n_map, Point(GRIDSIZE/2, GRIDSIZE/2), Size(GRIDSIZE/4, GRIDSIZE/4), 0, 0, 360, 0xff, 2, 8);
    
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
            x = ((int) (fx / stepgrid)) + (GRIDSIZE / 2);
            y = ((int) (fy / stepgrid)) + (GRIDSIZE / 2);
            printf("<%3.2f; %3.2f> \t <%d; %d> \t(%d, %d) \n", fx, fy, x, y, addr_x, addr_y);
            fp[0][j] = Point(x, y);
        }
        const Point* pts[1] = {fp[0]};
        int npt[] = {4};
        Scalar labelc;
        switch(labels[i])
        {
            case 1: // Vehicle
                labelc = Scalar(0xff, 0, 0, 0xff);
                break;

            case 2: // Pedestrian
                labelc = Scalar(0, 0, 0xff, 0xff);
                break;

            case 3: // Terrain
                labelc = Scalar(0, 0xff, 0, 0xff);
                break;

            default:
                labelc = Scalar(0, 0, 0, 0xff);
                break;
        }
        fillPoly( n_map, pts, npt, 1, labelc, LINE_8);
    }

    memcpy(map, n_map.ptr(), GRIDSIZE*GRIDSIZE*4);
    // delete n_map;
}