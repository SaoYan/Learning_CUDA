#include <stdio.h>
#include <cuda_runtime.h>

// reference
// https://devblogs.nvidia.com/maximizing-unified-memory-performance-cuda/

#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}                                                                            \

template <typename data_type>
__global__ void stream_thread(data_type *input, data_type *output, const int n);

int main(int argc, char **argv) {
    int n = 1<<20;
    size_t nBytes = n * sizeof(float);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // allocate unified memory
    float *in, *out;
    CHECK(cudaMallocManaged((float**)&in, nBytes));
    CHECK(cudaMallocManaged((float**)&out, nBytes));

    // pre-fetching
    // CHECK(cudaMemPrefetchAsync(in,  nBytes, dev, NULL));
    // CHECK(cudaMemPrefetchAsync(out, nBytes, dev, NULL));

    // launch kernel
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    printf("<<< %4d, %4d >>>\n", grid.x, block.x);
    stream_thread<float> <<<grid, block>>> (in, out, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // free memory
    CHECK(cudaFree(in));
    CHECK(cudaFree(out));
    
    // clean up all resources
    CHECK(cudaDeviceReset());
    return 0;
}

template <typename data_type>
__global__ void stream_thread(data_type *input, data_type *output, const int n) { 
    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    if (tid < n) {
        output[tid] = input[tid];
    }
}