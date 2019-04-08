#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}                                                                            \

__global__ void nestedHelloWorld(const int iSize, int iDepth);

int main(int argc, char **argv)
{
    int size = 8;
    int blockSize = 8; // initial block size
    int gridSize = 1;

    if(argc > 1) {
        gridSize = atoi(argv[1]);
        size = gridSize * blockSize;
    }

    dim3 block(blockSize);
    dim3 grid((size + block.x - 1) / block.x);
    printf("Execution Configuration: grid %d block %d\n", grid.x, block.x);

    nestedHelloWorld<<<grid, block>>>(block.x, 0);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());
    return 0;
}

__global__ void nestedHelloWorld(const int size, int depth) {
    printf("Recursion level %d: Hello World from thread %d block %d\n", depth, threadIdx.x, blockIdx.x);
    if (size == 1) return;

    // reduce block size to half
    int nthreads = size >> 1;

    // thread 0 launches child grid recursively
    if(threadIdx.x == 0 && nthreads > 0) {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, depth+1);
        printf("-------> nested execution depth: %d\n", depth);
    }
}
