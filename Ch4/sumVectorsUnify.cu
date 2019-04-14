#include <stdio.h>
#include <cuda_runtime.h>

void initialData(float *ip, const int N);
void sumArraysOnHost(float *A, float *B, float *C, const int N);
void verifyResult(float *hostRes, float *deviceRes, const int N);

__global__ void sumArraysOnDevice(float *A, float *B, float *C, const int N);

#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}                                                                            \

int main(int argc, char **argv) {
    int nElem = 1<<28;
    size_t nBytes = nElem * sizeof(float);
    printf("Vector size %d\n", nElem);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // allocate unified memory
    float *A, *B, *C, *C_gpu;
    CHECK(cudaMallocManaged((float**)&A, nBytes));
    CHECK(cudaMallocManaged((float**)&B, nBytes));
    CHECK(cudaMallocManaged((float**)&C, nBytes));
    CHECK(cudaMallocManaged((float**)&C_gpu, nBytes));

    // initialize data at host side
    initialData(A, nElem);
    initialData(B, nElem);
    memset(C, 0, nBytes);
    memset(C_gpu,  0, nBytes);

    // compute on CPU
    sumArraysOnHost(A, B, C, nElem);

    // launch CUDA kernel
    int threadPerBlock = 1024;
    dim3 block (threadPerBlock);
    dim3 grid((nElem + block.x - 1) / block.x);
    sumArraysOnDevice<<<grid, block>>>(A, B, C_gpu, nElem);

    // check results
    verifyResult(C, C_gpu, nElem);

    // free memory
    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(C));
    CHECK(cudaFree(C_gpu));

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

/**********CUDA kernels**********/

__global__ void sumArraysOnDevice(float *A, float *B, float *C, const int N) {
    // 1D grid of 1D block
    // compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

/**********host functions**********/

void initialData(float *ip, const int N) {
    // generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < N; i++) {
        ip[i] = (float) (rand() & 0xFF) / 10.f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

void verifyResult(float *hostRes, float *deviceRes, const int N) {
    double eps = 1e-8;
    for (int i = 0; i < N; i++) {
        if (abs(hostRes[i] - deviceRes[i] > eps)) {
            printf("Arrays do not match:\n");
            printf("host %5.2f gpu %5.2f at array index %d\n", hostRes[i], deviceRes[i], i);
            return;
        }
    }
    printf("Arrays match! Congrats, your kernel code works well!\n");
    return;
}
