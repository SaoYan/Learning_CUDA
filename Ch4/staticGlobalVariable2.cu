#include <stdio.h>
#include <cuda_runtime.h>

void printValue(float *ip, const int n);
__global__ void modifyGlobalVariable(const int n);

#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}                                                                            \

#define N 32

__device__ float devData[N];

int main(int argc, char **argv) {
    // initialize the global variable
    float *value = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        value[i] = 3.14f;
    }
    printValue(value, N);
    CHECK(cudaMemcpyToSymbol(devData, value, N * sizeof(float)));

    // invoke kernel
    modifyGlobalVariable<<<1, N>>>(N);
    CHECK(cudaDeviceSynchronize());
    
    // copy back to host
    CHECK(cudaMemcpyFromSymbol(value, devData, N * sizeof(float)));
    printValue(value, N);

    cudaDeviceReset();
    return 0;
}

__global__ void modifyGlobalVariable(const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        devData[idx] *= idx;
    }
} 

void printValue(float *ip, const int n) {
    for (int i = 0; i < n; i ++) {
        printf("%.2f, ", ip[i]);
    }
    printf("%c", '\n');
}
