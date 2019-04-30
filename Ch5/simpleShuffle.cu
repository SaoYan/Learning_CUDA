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

#define BDIMX 16
#define SEGM  4

void printData(int *data, const int size);



int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    int nElem = BDIMX;
    int h_in[BDIMX], h_out[BDIMX];
    for (int i = 0; i < nElem; i++) h_in[i] = i;
    printf("initialData\t\t: ");
    printData(h_in, nElem);

    size_t nBytes = nElem * sizeof(int);
    int *d_in, *d_out;
    CHECK(cudaMalloc((int**)&d_in, nBytes));
    CHECK(cudaMalloc((int**)&d_out, nBytes));
    CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));
    dim3 block(BDIMX);

    // Broadcasting a value across a warp
    cudaMemset(d_out, 0, nBytes);
    memset(h_out, 0, nBytes);
    shfl_broadcast<<<1, block>>>(d_out, d_in, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    printf("broadcast\t\t: ");
    printData(h_out, nElem);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaDeviceReset());
    return 0;
}

/**********host functions**********/

void printData(int *data, const int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
}

/**********CUDA kernels**********/

__global__ void shfl_broadcast(int *out, int*in, int srcLane) {
    int value = in[threadIdx.x];
    value = __shfl(value, srcLane, blockDim.x);
    out[threadIdx.x] = value;
}
