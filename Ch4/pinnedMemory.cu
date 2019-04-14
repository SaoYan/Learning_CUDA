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

int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    if (!deviceProp.canMapHostMemory) {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    int size = 1 << 30;
    size_t nBytes = size * sizeof(float);
    clock_t start, end;
    double time;

    // allocate host memory
    float *h_a, *h_a_pin;
    h_a = (float *) malloc(nBytes);
    CHECK(cudaMallocHost((float **)&h_a_pin, nBytes)); // pinned memory
    memset(h_a, 0, nBytes);
    memset(h_a_pin, 0, nBytes);

    // allocate device memory
    float *d_a1, *d_a2;
    CHECK(cudaMalloc((float **)&d_a1, nBytes));
    CHECK(cudaMalloc((float **)&d_a2, nBytes));

    // pageable memory <--> device
    start = clock();
    CHECK(cudaMemcpy(d_a1, h_a, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(h_a, d_a1, nBytes, cudaMemcpyDeviceToHost));
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("data transfer between pageable memory and device takes %.4f ms\n", time * 1000);

    // pinned memory <--> device
    start = clock();
    CHECK(cudaMemcpy(d_a2, h_a_pin, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(h_a_pin, d_a2, nBytes, cudaMemcpyDeviceToHost));
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("data transfer between pinned memory and device takes %.4f ms\n", time * 1000);

    free(h_a);
    CHECK(cudaFreeHost(h_a_pin));
    CHECK(cudaFree(d_a1));
    CHECK(cudaFree(d_a2));

    CHECK(cudaDeviceReset());
    return 0;
}