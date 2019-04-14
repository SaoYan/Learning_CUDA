#include <stdio.h>
#include <cuda_runtime.h>

void initialData(float *ip, const int N);
void sumArraysOnHost(float *A, float *B, float *C, const int N);
void verifyResult(float *hostRes, float *deviceRes, const int N);

__global__ void sumArraysOnDevice(float *A, float *B, float *C, const int N);
__global__ void sumArraysZeroCopy(float *A, float *B, float *C, const int N);

#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}                                                                            \

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // check if support mapped memory
    if (!deviceProp.canMapHostMemory) {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    // set up data size of vectors
    int ipower = 10;
    if (argc > 1) ipower = atoi(argv[1]);
    int nElem = 1 << ipower;
    size_t nBytes = nElem * sizeof(float);
    if (ipower < 18) {
        printf("Vector length: %d, data size:  %3.0f KB\n", nElem, (float)nBytes / 1024.0f);
    }
    else {
        printf("Vector length: %d data size  %3.0f MB\n", nElem, (float)nBytes / (1024.0f * 1024.0f));
    }

    // part 1: using device memory
    // malloc host memory
    float *h_A, *h_B, *h_C, *h_C_gpu;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    h_C     = (float *)malloc(nBytes);
    h_C_gpu = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(h_C, 0, nBytes);
    memset(h_C_gpu,  0, nBytes);

    // compute on CPU
    sumArraysOnHost(h_A, h_B, h_C, nElem);

    // allocate device memory (global memory)
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from CPU to GPU
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // launch CUDA kernel
    int threadPerBlock = 1024;
    dim3 block (threadPerBlock);
    dim3 grid((nElem + block.x - 1) / block.x);
    sumArraysOnDevice<<<grid, block>>>(d_A, d_B, d_C, nElem);

    // copy data from GPU back to CPU
    CHECK(cudaMemcpy(h_C_gpu, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check results
    verifyResult(h_C, h_C_gpu, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));

    // free host memory
    free(h_A);
    free(h_B);

    // part 2: using zerocopy memory for array A and B
    // allocate zerocpy memory
    CHECK(cudaHostAlloc((void **)&h_A, nBytes, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void **)&h_B, nBytes, cudaHostAllocMapped));

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(h_C, 0, nBytes);
    memset(h_C_gpu,  0, nBytes);

    // pass the pointer to device
    // no need when using unified virtual addressing (UVA)
    // CHECK(cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0));
    // CHECK(cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0));

    // compute on CPU
    sumArraysOnHost(h_A, h_B, h_C, nElem);

    // execute kernel with zero copy memory
    // sumArraysZeroCopy<<<grid, block>>>(d_A, d_B, d_C, nElem);
    sumArraysZeroCopy<<<grid, block>>>(h_A, h_B, d_C, nElem); // UVA

    // copy data from GPU back to CPU
    CHECK(cudaMemcpy(h_C_gpu, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    verifyResult(h_C, h_C_gpu, nElem);

    // free memory
    CHECK(cudaFree(d_C));
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
    free(h_C);
    free(h_C_gpu);

    // cannot free d_A and d_B because they are already freed
    // remember: in part 2 d_A and d_B refer to the same memory as h_A and h_B

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

__global__ void sumArraysZeroCopy(float *A, float *B, float *C, const int N) {
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
