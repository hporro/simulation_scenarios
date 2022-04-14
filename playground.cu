#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>

//declare constant memory

struct angle {
    float a;
    int b;
};

__constant__ angle cangle;


__global__ void test_kernel(float* darray)
{
    int index;

    //calculate each thread global index
    index = blockIdx.x * blockDim.x + threadIdx.x;

#pragma unroll 10
    for (int loop = 0; loop < 360; loop++)
        darray[index] = darray[index] + cangle.a;
    return;

}

int main(int argc, char** argv)
{
    int size = 3200;
    float* darray;
    float hangle[360];

    //allocate device memory
    cudaMalloc((void**)&darray, sizeof(float) * size);

    //initialize allocated memory
    cudaMemset(darray, 0, sizeof(float) * size);

    //initialize angle array on host
    for (int loop = 0; loop < 360; loop++)
        hangle[loop] = acos(-1.0f) * loop / 180.0f;

    //copy host angle data to constant memory
    //cudaMemcpyToSymbol(&cangle, &hangle[0], sizeof(float));

    test_kernel <<<size/64,64>>> (darray);

    //free device memory
    cudaFree(darray);
    return 0;
}
