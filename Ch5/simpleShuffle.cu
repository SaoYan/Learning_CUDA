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

#define BDIMX 32
#define SEGM  4
#define MASK 0xffffffff

void printData(int *data, const int size);

__global__ void shfl_broadcast(int *out, int *in, const int srcLane);
__global__ void shfl_up(int *out, int *in, const int offset);
__global__ void shfl_down(int *out, int *in, const int offset);
__global__ void shfl_around(int *out, int *in, const int offset);
__global__ void shfl_xor(int *out, int *in, const int laneMask);
__global__ void shfl_xor_array(int *out, int *in, const int laneMask);
__global__ void shfl_swap(int *out, int *in, const int laneMask, int index);

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

    // Shift up within a warp
    cudaMemset(d_out, 0, nBytes);
    memset(h_out, 0, nBytes);
    shfl_up<<<1, block>>>(d_out, d_in, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    printf("shift up\t\t: ");
    printData(h_out, nElem);

    // Shift down within a warp
    cudaMemset(d_out, 0, nBytes);
    memset(h_out, 0, nBytes);
    shfl_down<<<1, block>>>(d_out, d_in, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    printf("shift down\t\t: ");
    printData(h_out, nElem);

    // Shift down around
    cudaMemset(d_out, 0, nBytes);
    memset(h_out, 0, nBytes);
    shfl_around<<<1, block>>>(d_out, d_in, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    printf("shift down around\t: ");
    printData(h_out, nElem);

    // Butterfly exchange  
    cudaMemset(d_out, 0, nBytes);
    memset(h_out, 0, nBytes);
    shfl_xor<<<1, block>>>(d_out, d_in, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    printf("butterfly exchange\t: ");
    printData(h_out, nElem);

    // Butterfly exchange array  
    cudaMemset(d_out, 0, nBytes);
    memset(h_out, 0, nBytes);
    shfl_xor_array<<<1, block.x / SEGM>>>(d_out, d_in, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    printf("butterfly exchange\t: ");
    printData(h_out, nElem);

    // Exchange values using array indices
    cudaMemset(d_out, 0, nBytes);
    memset(h_out, 0, nBytes);
    shfl_swap<<<1, block.x / SEGM>>>(d_out, d_in, 1, 3);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    printf("value exchange\t\t: ");
    printData(h_out, nElem);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaDeviceReset());
    return 0;
}

/**********host functions**********/

void printData(int *data, const int size) {
    for (int i = 0; i < size; i++) {
        printf("%2d ", data[i]);
    }
    printf("\n");
}

/**********CUDA kernels**********/

__global__ void shfl_broadcast(int *out, int *in, const int srcLane) {
    int value = in[threadIdx.x];
    value = __shfl_sync(MASK, value, srcLane, blockDim.x);
    out[threadIdx.x] = value;
}

__global__ void shfl_up(int *out, int *in, const int offset) {
    int value = in[threadIdx.x];
    value = __shfl_up_sync(MASK, value, offset, blockDim.x);
    out[threadIdx.x] = value;
}

__global__ void shfl_down(int *out, int *in, const int offset) {
    int value = in[threadIdx.x];
    value = __shfl_down_sync(MASK, value, offset, blockDim.x);
    out[threadIdx.x] = value;
}

__global__ void shfl_around(int *out, int *in, const int offset) {
    int value = in[threadIdx.x];
    value = __shfl_sync(MASK, value, threadIdx.x + offset, blockDim.x);
    out[threadIdx.x] = value;
}

__global__ void shfl_xor(int *out, int *in, const int laneMask) {
    int value = in[threadIdx.x];
    value = __shfl_xor_sync(MASK, value, laneMask, blockDim.x);
    out[threadIdx.x] = value;
}

__global__ void shfl_xor_array(int *out, int *in, const int laneMask) {
    int idx = threadIdx.x * SEGM;
    int value[SEGM];
    for (int i = 0; i < SEGM; i++) value[i] = in[idx + i];
    
    value[0] = __shfl_xor_sync(MASK, value[0], laneMask, BDIMX); 
    value[1] = __shfl_xor_sync(MASK, value[1], laneMask, BDIMX); 
    value[2] = __shfl_xor_sync(MASK, value[2], laneMask, BDIMX); 
    value[3] = __shfl_xor_sync(MASK, value[3], laneMask, BDIMX);

    for (int i = 0; i < SEGM; i++) out[idx + i] = value[i];
}

__global__ void shfl_swap(int *out, int *in, const int laneMask, int index) {
    int idx = threadIdx.x * SEGM;
    int value[SEGM];
    for (int i = 0; i < SEGM; i++) value[i] = in[idx + i];

    value[index] = __shfl_xor_sync(MASK, value[index], laneMask, BDIMX);

    for (int i = 0; i < SEGM; i++) out[idx + i] = value[i];
}
