#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#define VEHICLE_MASK        0b01000000
#define PEDESTRIAN_MASK     0b10000000
#define TERRAIN_MASK        0b00000010

void bonjour_cpp();
void mean_merger_cpp(unsigned char *masks, int gridsize, int n_agents, float *out);
void set_inter(const char *A, const char *B, char *out);
void set_union(const char *A, const char *B, char *out);
bool set_cmp(const char *A, const char *B);


extern "C" {
    void bonjour()
        {bonjour_cpp();}
    void mean_merger(unsigned char *masks, int gridsize, int n_agents, float *out)
        {mean_merger_cpp(masks, gridsize, n_agents, out);}
    //     {mean_merger_cpp(masks, gridsize, out);}
}

using namespace std;

int main(int argc, char **argv)
{
    char FE[8][4] = {"O", "V", "P", "T", "VP", "VT", "PT", "VPT"};
    char out[4] = {0};
    unsigned char mask[5][600][600] = {0};
    float outt[600][600][3] = {0};

    mean_merger_cpp((unsigned char *) mask, 600, 5, (float *) outt);
    set_union(FE[4], FE[5], out);

    printf("%s\n", out);

    cout << set_cmp("VP", "PVT") << endl;

    

    return 0;
}

void bonjour_cpp()
{
    printf("Bonjour!!!\n");
}

void set_inter(const char *A, const char *B, char *out)
{
    int i = 0, j = 0;
    for(i=0; i<strlen(A); i++)
    {
        if(strchr(B, A[i]) != NULL)
        {
            out[j] = A[i];
            j++;
        }
    }   
}

void set_union(const char *A, const char *B, char *out)
{
    int i = 0, j = strlen(B);
    out = strcpy(out, B);
    for(i=0; i<strlen(A); i++)
    {
        if(strchr(B, A[i]) == NULL)
        {
            out[j] = A[i];
            j++;
        }
    }   
}

bool set_cmp(const char *A, const char *B)
{
    int i = 0;
    if(strlen(A) != strlen(B))
        return false;

    for(i = 0; i<strlen(A); i++)
    {
        if(strchr(B, A[i]) == NULL)
            return false;
    }
    return true;
}

void mean_merger_cpp(unsigned char *masks, int gridsize, int n_agents, float *out)
{
    int l = 0, i = 0, c = 0;
    int idx = 0;
    for(l = 0; l<n_agents; l++)
    {
        for(i=0; i<gridsize*gridsize; i++)
        {
            switch(masks[l*(gridsize*gridsize) + i])
            {
                case VEHICLE_MASK:
                    out[(i*3) + 0] += 1.0;
                    out[(i*3) + 1] += 0.0;
                    out[(i*3) + 2] += 0.0;
                    break;

                case PEDESTRIAN_MASK:
                    out[(i*3) + 0] += 0.0;
                    out[(i*3) + 1] += 1.0;
                    out[(i*3) + 2] += 0.0;
                    break;

                case TERRAIN_MASK:
                    out[(i*3) + 0] += 0.0;
                    out[(i*3) + 1] += 0.0;
                    out[(i*3) + 2] += 1.0;
                    break;
                
                default:
                    out[(i*3) + 0] += 0.5;
                    out[(i*3) + 1] += 0.5;
                    out[(i*3) + 2] += 0.5;
                    break;
            }
        }
    }
    for(i = 0; i<(gridsize*gridsize*3); i++)
        out[i] /= n_agents;
}