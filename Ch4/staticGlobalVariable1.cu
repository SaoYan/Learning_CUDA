#include <stdio.h>
#include <cuda_runtime.h>

__global__ void checkGlobalVariable();

#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}                                                                            \

__device__ float devData;

int main(int argc, char **argv) {
    // initialize the global variable
    float value = 3.14f;
    CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float)));
    printf("Host:   copied %f to the global variable\n", value);

    // invoke kernel
    checkGlobalVariable<<<1, 32>>>();
    
    // copy back to host
    CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
    printf("Host:   the value changed by the kernel to %f\n", value);

    cudaDeviceReset();
    return 0;
}

__global__ void checkGlobalVariable() {
    printf("Device: the value of the global variable is %f\n", devData);
    devData += 2.0f;
    __syncthreads();
} 
