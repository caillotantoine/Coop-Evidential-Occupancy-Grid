#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#define VEHICLE_MASK        0b01000000
#define PEDESTRIAN_MASK     0b10000000
#define TERRAIN_MASK        0b00000010

#define DEMPSTER            0x00
#define CONJUNCTIVE         0x01
#define DISJUNCTIVE         0x02

#define CUDA_BLOCKWIDTH     (256)
#define N_CLASSES           8

// CPP function to be available on library front end
void bonjour_cpp();
void mean_merger_cpp(unsigned char *masks, int gridsize, int n_agents, float *out);
void DST_merger_CPP(float *evid_maps_in, float *inout, int gridsize, int nFE, int n_agents, unsigned char method);
void DST_merger_CUDA_CPP(float *evid_maps_in, float *inout, int gridsize, int nFE, int n_agents, unsigned char method);

// Merging functions, either for CUDA and CPP
__host__ __device__ void conjunctive(float *inout_cell, float *cell, int n_elem, bool dempster);
__host__ __device__ void disjunctive(float *inout_cell, float *cell, int n_elem);
__host__ __device__ float Konflict(float *inout_cell, float *cell, int n_elem);

// Obsolete function? 
void set_inter(const char *A, const char *B, char *out);
void set_union(const char *A, const char *B, char *out);
bool set_cmp(const char *A, const char *B);

// Interface of the SO library
extern "C" {
    void bonjour()
        {bonjour_cpp();}
    void mean_merger(unsigned char *masks, int gridsize, int n_agents, float *out)
        {mean_merger_cpp(masks, gridsize, n_agents, out);}
    void DST_merger(float *evid_maps_in, float *inout, int gridsize, int nFE, int n_agents, unsigned char method)
        {DST_merger_CPP(evid_maps_in, inout, gridsize, nFE, n_agents, method);}
    void DST_merger_CUDA(float *evid_maps_in, float *inout, int gridsize, int nFE, int n_agents, unsigned char method)
        {DST_merger_CUDA_CPP(evid_maps_in, inout, gridsize, nFE, n_agents, method);}
}

//////////////////////////
//                      //
//     CUDA kernels     //
//                      //
//////////////////////////

// TODO : Doesn't work. Seem to only process the first agent, thus, no merging done. 
//   NOTE : I certainly messed up with memory management in somewhere.

__global__ void conjunctive_kernel(float *evid_maps_in, float *inout, const int gridsize, const int nFE, const int n_agents)
{
    const long i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = 0;
    if(i < (gridsize * gridsize))
    {
        for(j = 0; j < n_agents; j++)
        {
            conjunctive((inout + i*nFE), (evid_maps_in + i*nFE*n_agents + j*nFE), nFE, false);
        }
    }
}

__global__ void dempster_kernel(float *evid_maps_in, float *inout, const int gridsize, const int nFE, const int n_agents)
{
    const long i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = 0;
    if(i < (gridsize * gridsize))
    {
        for(j = 0; j < n_agents; j++)
            conjunctive((inout + i*nFE), (evid_maps_in + i*nFE*n_agents + j*nFE), nFE, true);
    }
}

__global__ void disjunctive_kernel(float *evid_maps_in, float *inout, const int gridsize, const int nFE, const int n_agents)
{
    const long i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = 0;
    if(i < (gridsize * gridsize))
    {
        for(j = 0; j < n_agents; j++)
            disjunctive((inout + i*nFE), (evid_maps_in + i*nFE*n_agents + j*nFE), nFE);
    }
}


///////////////////////////
//                       //
//   Merging functions   //
//                       //
///////////////////////////

__host__ __device__ void conjunctive(float *inout_cell, float *cell, int n_elem, bool dempster)
{
    int A = 0, B = 0, C = 0, i = 0;
    float buf[N_CLASSES] = {0};
    float res;
    float K = 0.0;
    if(dempster)
        K = 1.0 / (1.0 - Konflict(inout_cell, cell, n_elem));
    for (A = 0; A<n_elem; A++)
    {
        for(B=0; B<n_elem; B++)
        {
            for(C=0; C<n_elem; C++)
            {
                if((B&C) == A)
                {
                    res = (float) *(inout_cell + B) * (float) *(cell + C);
                    // printf("A %f, %f, %f\n",*(inout_cell + B), *(cell + C), res);
                    buf[A] += res;
                }
            }
        }
        if(dempster)
            buf[A] *= K;
    }
    for(i = 0; i<N_CLASSES; i++)
        inout_cell[i] = buf[i];
    // memcpy(inout_cell, buf, sizeof(float)*n_elem);
}

__host__ __device__ void disjunctive(float *inout_cell, float *cell, int n_elem)
{
    int A = 0, B = 0, C = 0, i=0;
    float buf[N_CLASSES] = {0};
    float res;
    for (A = 0; A<n_elem; A++)
    {
        for(B=0; B<n_elem; B++)
        {
            for(C=0; C<n_elem; C++)
            {
                if((B|C) == A)
                {
                    res = (float) *(inout_cell + B) * (float) *(cell + C);
                    // printf("A %f, %f, %f\n",*(inout_cell + B), *(cell + C), res);
                    buf[A] += res;
                }
            }
        }
    }
    for(i = 0; i<N_CLASSES; i++)
        inout_cell[i] = buf[i];
    // memcpy(inout_cell, buf, sizeof(float)*n_elem);
}

__host__ __device__ float Konflict(float *inout_cell, float *cell, int n_elem)
{
    int B = 0, C = 0;
    float res = 0;
    for(B=0; B<n_elem; B++)
    {
        for(C=0; C<n_elem; C++)
        {
            if((B|C) == 0)
            {
                res += (float) *(inout_cell + B) * (float) *(cell + C);
            }
        }
    }
    return res;
}


/////////////////////////
//                     //
//     Entry point     //
//                     //
/////////////////////////

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

//////////////////////////
//                      //
//       Functions      //
//                      //
//////////////////////////

// Test - obsolete
void bonjour_cpp()
{
    printf("Bonjour!!!\n");
}

// Obsolete
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

// Obsolete
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

// Obsolete
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

// Merger code for averaging cells

void mean_merger_cpp(unsigned char *masks, int gridsize, int n_agents, float *out)
{
    int l = 0, i = 0, c = 0;
    int idx = 0;
    for(i=0; i<gridsize*gridsize; i++)
    {
        for(l=0; l<n_agents; l++)
        {
            switch(masks[i*n_agents + l])
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
        out[i] /= n_agents; ////
}

// Merger function to merge the cells using Dempster Shaffer Theory 

void DST_merger_CPP(float *evid_maps_in, float *inout, int gridsize, int nFE, int n_agents, unsigned char method)
{
    int l = 0, i = 0, j =0;
    for(i=0; i<gridsize*gridsize; i++)
    {
        for(j = 0; j<n_agents; j++)
        {
            //inout[i*nFE /*+ channel index*/] = evid_maps_in[i*nFE*n_agents + j*nFE /*+ channel index*/];
            switch(method)
            {
                case CONJUNCTIVE:
                    conjunctive((inout + i*nFE), (evid_maps_in + i*nFE*n_agents + j*nFE), nFE, false);
                    break;
                
                case DISJUNCTIVE:
                    disjunctive((inout + i*nFE), (evid_maps_in + i*nFE*n_agents + j*nFE), nFE);
                    break;

                case DEMPSTER:
                    conjunctive((inout + i*nFE), (evid_maps_in + i*nFE*n_agents + j*nFE), nFE, true);
                    break;

                default:
                    printf("No fusion method for the following value: %d", method);
                    break;
            }
        }   
    }

}

// Merger function to merge the cells using Dempster Shaffer Theory with CUDA

void DST_merger_CUDA_CPP(float *evid_maps_in, float *inout, int gridsize, int nFE, int n_agents, unsigned char method)
{
    float *dev_evid_map = NULL;
    float *dev_inout = NULL;

    const int gridwidth_d1 = 1 + (((gridsize*gridsize)-1) / CUDA_BLOCKWIDTH);
    const dim3 gridwidth(gridwidth_d1, 1, 1);
    const dim3 blockwidth(CUDA_BLOCKWIDTH, 1, 1);

    cudaMalloc(&dev_evid_map, sizeof(float)*gridsize*gridsize*n_agents*nFE);
    cudaMalloc(&dev_inout, sizeof(float)*gridsize*gridsize*nFE);

    cudaMemcpy(dev_evid_map, evid_maps_in, sizeof(float)*gridsize*gridsize*n_agents*nFE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_inout, inout, sizeof(float)*gridsize*gridsize*nFE, cudaMemcpyHostToDevice);


    switch(method)
    {
        case CONJUNCTIVE:
            conjunctive_kernel <<<gridwidth, blockwidth>>> (evid_maps_in, inout, gridsize, nFE, n_agents);
            break;
        
        case DISJUNCTIVE:
            disjunctive_kernel <<<gridwidth, blockwidth>>> (evid_maps_in, inout, gridsize, nFE, n_agents);
            break;

        case DEMPSTER:
            dempster_kernel <<<gridwidth, blockwidth>>> (evid_maps_in, inout, gridsize, nFE, n_agents);
            break;

        default:
            printf("No fusion method for the following value: %d", method);
            break;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(evid_maps_in, dev_evid_map, sizeof(float)*gridsize*gridsize*n_agents*nFE, cudaMemcpyDeviceToHost);
    cudaMemcpy(inout, dev_inout, sizeof(float)*gridsize*gridsize*nFE, cudaMemcpyDeviceToHost);

    cudaFree(dev_evid_map);
    cudaFree(dev_inout);
}



