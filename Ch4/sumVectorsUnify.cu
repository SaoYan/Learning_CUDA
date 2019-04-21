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
    int ipower = 28;
    if (argc > 1) ipower = atoi(argv[1]);
    int nElem = 1<<ipower;
    size_t nBytes = nElem * sizeof(float);
    clock_t start, end;
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
    memset(C_gpu, 0, nBytes);

    // compute on CPU
    start = clock();
    sumArraysOnHost(A, B, C, nElem);
    end = clock();
    double cpuTime = ((double) (end - start)) / CLOCKS_PER_SEC;

    // prefetching
    int device = -1;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaMemPrefetchAsync(A, nBytes, device, NULL));
    CHECK(cudaMemPrefetchAsync(B, nBytes, device, NULL));
    CHECK(cudaMemPrefetchAsync(C_gpu, nBytes, device, NULL));

    // launch CUDA kernel
    int threadPerBlock = 1024;
    if (argc > 2) ipower = atoi(argv[2]);
    dim3 block (threadPerBlock);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("Grid dimension %d Block dimensiton %d\n",grid.x, block.x);
    start = clock();
    sumArraysOnDevice<<<grid, block>>>(A, B, C_gpu, nElem);
    CHECK(cudaDeviceSynchronize()); // synchronization is necessary when using unified memory!
    end = clock();
    double gpuTime = ((double) (end - start)) / CLOCKS_PER_SEC;

    // check results
    verifyResult(C, C_gpu, nElem);
    printf("It takes %.4f sec to execute on CPU\n", cpuTime);
    printf("It takes %.4f sec to execute on GPU\n", gpuTime);

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
        if (abs(hostRes[i] - deviceRes[i]) > eps) {
            printf("Arrays do not match:\n");
            printf("host %5.2f gpu %5.2f at array index %d\n", hostRes[i], deviceRes[i], i);
            return;
        }
    }
    printf("Arrays match! Congrats, your kernel code works well!\n");
    return;
}
