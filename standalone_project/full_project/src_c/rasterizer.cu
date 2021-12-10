#include <stdio.h>
#include <stdlib.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAPSIZE (120.0)
#define GRIDSIZE (120*5)

#define NCLASS              4

#define VEHICLE_MASK        0b01000000
#define PEDESTRIAN_MASK     0b10000000
#define TERRAIN_MASK        0b00000010


void bonjour_cpp();
void projector_cpp(int len, int* labels, float* fp_vec, unsigned char* map, float mapsize, int gridsize);
unsigned char* test_read_write_cpp (int len, float *fp_poly, int *label, unsigned char *out);
void apply_BBA_cpp(const int nFE, const int gridsize, float *FE, unsigned char *map, float *evid_map, bool CUDA);

extern "C" {
    void bonjour()
        {bonjour_cpp();}
    void projector(int len, int* labels, float* fp_vec, unsigned char* map, float mapsize, int gridsize) 
        {projector_cpp(len, labels, fp_vec, map, mapsize, gridsize);}
    unsigned char* test_read_write (int len, float *fp_poly, int *label, unsigned char *out)
        {return test_read_write_cpp (len, fp_poly, label, out);}
    void apply_BBA(const int nFE, const int gridsize, float *FE, unsigned char *map, float *evid_map)
        {apply_BBA_cpp(nFE, gridsize, FE, map, evid_map, false);}
}

using namespace cv;

int main(int argc, char **argv)
{
    float fp_vec[2][4][2] = {{{7.73851759, -49.70380452}, {4.39891667, -49.70377771}, {3.45951359, -38.3454241}, {6.0859267,  -38.34544071}},
                             {{2.70641359, -48.17095769}, {0.8450105,  -48.17094317}, {0.72571429, -40.87830044}, {2.3243211,  -40.87831116}}};


    int label[2] = {1, 2};
    const int nFE = 4;

    Mat map = Mat::zeros(GRIDSIZE, GRIDSIZE, CV_8U);
    Mat evidmap = Mat::zeros(GRIDSIZE, GRIDSIZE, CV_32FC4);
    //                        V    P    T    VPT
    float FE[NCLASS][nFE] = {{0.1, 0.1, 0.1, 0.7},
                             {0.7, 0.1, 0.1, 0.1}, 
                             {0.1, 0.7, 0.1, 0.1}, 
                             {0.1, 0.1, 0.7, 0.1}};

    int i;
    for (i = 0; i<4; i++)
    {
        printf("<%3.2f; %3.2f>\n", fp_vec[0][i][0], fp_vec[0][i][1]);
    }
    for (i = 0; i<4; i++)
    {
        printf("<%3.2f; %3.2f>\n", fp_vec[1][i][0], fp_vec[1][i][1]);
    }
    // printf("test nvcc!!!!!\n");
    // ellipse(map, Point(GRIDSIZE/4, GRIDSIZE/4), Size(GRIDSIZE/4, GRIDSIZE/4), 0, 0, 360, 0xff, 2, 8);
    projector_cpp(2, label, (float*) fp_vec, map.ptr(), MAPSIZE, GRIDSIZE);
    apply_BBA_cpp(nFE, GRIDSIZE, (float *) FE, map.ptr(), (float *) evidmap.ptr(), false);

    imshow("Test map display", map);
    waitKey(0);
    imshow("Test map display", evidmap);
    waitKey(0);
    return 0;
}

void bonjour_cpp()
{
    printf("Bonjour!!!\n");
}

void projector_cpp(const int len, int* labels, float* fp_vec, unsigned char* map, float mapsize, int gridsize)
{
    int i, j;
    // Mat n_map = Mat(gridsize, gridsize, CV_8U, map);
    Mat n_map = Mat::zeros(gridsize, gridsize, CV_8U);
    Point fp[1][4];
    float fx, fy;
    int x, y;
    float *addr_x, *addr_y;
    float stepgrid = ((float) mapsize) / gridsize;

    // ellipse(n_map, Point(GRIDSIZE/2, GRIDSIZE/2), Size(GRIDSIZE/4, GRIDSIZE/4), 0, 0, 360, 0xff, 2, 8);
    
    // printf("%3.2f\n", *(fp_vec + 3));
    for(i=0; i<len; i++)
    {
        for(j=0; j<4; j++)
        {
            // printf("Label %d: ", labels[i]);
            addr_x = (fp_vec + 8*i + 2*j);
            addr_y = (fp_vec + 8*i + 2*j + 1);
            fx = *addr_x;
            fy = *addr_y;
            x = ((int) (fx / stepgrid)) + (gridsize / 2);
            y = ((int) (fy / stepgrid)) + (gridsize / 2);
            // printf("<%3.2f; %3.2f> \t <%d; %d> \t(%d, %d) \n", fx, fy, x, y, addr_x, addr_y);
            fp[0][j] = Point(x, y);
        }
        const Point* pts[1] = {fp[0]};
        int npt[] = {4};
        Scalar labelc;
        switch(labels[i])
        {
            case 1: // Vehicle
                labelc = VEHICLE_MASK;
                break;

            case 2: // Pedestrian
                labelc = PEDESTRIAN_MASK;
                break;

            case 3: // Terrain
                labelc = TERRAIN_MASK;
                break;

            default:
                labelc = 0x00;
                break;
        }
        fillPoly( n_map, pts, npt, 1, labelc, LINE_8);
    }

    memcpy(map, n_map.ptr(), gridsize*gridsize);
    // delete n_map;
}

unsigned char* test_read_write_cpp (int len, float *fp_poly, int *label, unsigned char *out)
{
    int idx = 0;
    Mat map = Mat::zeros(GRIDSIZE, GRIDSIZE, CV_8U);
    for(idx = 0; idx < len; idx++)
    {
        printf("%03d: \t%d\n", idx, label[idx]);
    }
    projector_cpp(len, label, fp_poly, map.ptr(), MAPSIZE, GRIDSIZE);

    // imshow("Test map display", map);
    // waitKey(0);
    memcpy(out, map.ptr(), GRIDSIZE*GRIDSIZE);
    // out = map.ptr();
    return map.ptr();
}

void apply_BBA_cpp(const int nFE, const int gridsize, float *FE, unsigned char *map, float *evid_map, bool CUDA)
{
    float *evidmap = NULL;
    evidmap = new float[gridsize * gridsize * nFE];
    unsigned int i = 0, j = 0;
    unsigned char cell = 0;

    for(i = 0; i<(gridsize*gridsize); i++)
    {
        // branchless switch
        cell = 0;
        cell += ((map[i] & VEHICLE_MASK) == VEHICLE_MASK) * 1;
        cell += ((map[i] & PEDESTRIAN_MASK) == PEDESTRIAN_MASK) * 2;
        cell += ((map[i] & TERRAIN_MASK) == TERRAIN_MASK) * 3;
        for(j = 0; j<nFE; j++)
        {
            *(evidmap + i*nFE + j) = *(FE + cell * nFE + j);
        }
    }
    memcpy(evid_map, evidmap, sizeof(float)*gridsize*gridsize*nFE);
}