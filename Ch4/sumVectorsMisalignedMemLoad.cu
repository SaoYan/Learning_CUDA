#include <stdio.h>
#include <cuda_runtime.h>

void initialData(float *ip, const int N);
void sumArraysOnHost(float *A, float *B, float *C, const int N, const int offset);
void verifyResult(float *hostRes, float *deviceRes, const int N);

__global__ void warmup(float *A, float *B, float *C, const int N, const int offset);
__global__ void sumArraysOnDeviceOffset(float *A, float *B, float *C, const int N, const int offset);
__global__ void sumArraysOnDeviceOffsetUnroll2(float *A, float *B, float *C, const int n, const int offset);
__global__ void sumArraysOnDeviceOffsetUnroll4(float *A, float *B, float *C, const int n, const int offset);

__global__ void sumArraysReadonlyCache(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, const int N, const int offset);

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
    clock_t start, end;
    double time;
    printf("Vector size %d\n", nElem);

    int offset = 0;
    if (argc>1) offset = atoi(argv[1]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // allocate host memory
    float *h_A, *h_B, *h_C, *h_C_gpu;
    h_A = (float *) malloc(nBytes);
    h_B = (float *) malloc(nBytes);
    h_C = (float *) malloc(nBytes);
    h_C_gpu = (float *) malloc(nBytes);

    // initial data (in CPU mem)
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(h_C, 0, nBytes);

    // compute on CPU
    // start = clock();
    // sumArraysOnHost(h_A, h_B, h_C, nElem, offset);
    // end = clock();
    // time = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("CPU execution: %.4f ms\n", time * 1000);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from CPU to GPU
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // configuration
    dim3 block(1024);
    dim3 grid((nElem+block.x-1)/block.x);
    printf("Grid dimension %d Block dimensiton %d Memory offset %d\n", grid.x, block.x, offset);

    // 0. warm up
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    CHECK(cudaGetLastError());

    // 1. no unrolling
    cudaMemset(d_C, 0, nBytes);
    memset(h_C_gpu, 0, nBytes);
    start = clock();
    sumArraysOnDeviceOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    // sumArraysReadonlyCache<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("GPU execution without unrolling: %.4f ms\n", time * 1000);
    // check result
    CHECK(cudaMemcpy(h_C_gpu, d_C, nBytes, cudaMemcpyDeviceToHost));
    verifyResult(h_C, h_C_gpu, nElem);
    printf("%f, %f", h_C[0], h_C_gpu[0]);

    // 2. x2 unrolling
    cudaMemset(d_C, 0, nBytes);
    memset(h_C_gpu, 0, nBytes);
    start = clock();
    sumArraysOnDeviceOffsetUnroll2<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("GPU execution x2 unrolling:      %.4f ms\n", time * 1000);
    // check result
    CHECK(cudaMemcpy(h_C_gpu, d_C, nBytes, cudaMemcpyDeviceToHost));
    verifyResult(h_C, h_C_gpu, nElem);

    // 3. x4 unorlling
    cudaMemset(d_C, 0, nBytes);
    memset(h_C_gpu, 0, nBytes);
    start = clock();
    sumArraysOnDeviceOffsetUnroll4<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("GPU execution x4 unrolling:      %.4f ms\n", time * 1000);
    // check result
    CHECK(cudaMemcpy(h_C_gpu, d_C, nBytes, cudaMemcpyDeviceToHost));
    verifyResult(h_C, h_C_gpu, nElem);

    // free host mem
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_gpu);

    // free device mem
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    
    // clean up all resources
    CHECK(cudaDeviceReset());
    return 0;
}

/**********CUDA kernels**********/

__global__ void warmup(float *A, float *B, float *C, const int N, const int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = idx + offset;
    if (k < N) C[idx] = A[k] + B[k];
}

__global__ void sumArraysOnDeviceOffset(
    float *A, float *B, float *C, 
    const int N, const int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = idx + offset;
    if (k < N) C[idx] = A[k] + B[k];
}

__global__ void sumArraysOnDeviceOffsetUnroll2(
    float *A, float *B, float *C, 
    const int N, const int offset) {
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int k = i + offset;
    
    if (k < N) C[i] = A[k] + B[k];
    if (k + blockDim.x < N) {
        C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
    }
}

__global__ void sumArraysOnDeviceOffsetUnroll4(
    float *A, float *B, float *C, 
    const int N, const int offset) {
    int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int k = i + offset;
    
    if (k < N) C[i] = A[k] + B[k];
    if (k + blockDim.x < N) {
        C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
    }
    if (k + 2 * blockDim.x < N) {
        C[i + 2 * blockDim.x] = A[k + 2 * blockDim.x] + B[k + 2 * blockDim.x];
    }
    if (k + 3 * blockDim.x < N) {
        C[i + 3 * blockDim.x] = A[k + 3 * blockDim.x] + B[k + 3 * blockDim.x];
    }
}

__global__ void sumArraysReadonlyCache(
    const float * __restrict__ A, 
    const float * __restrict__ B, 
    float * __restrict__ C, 
    const int N, const int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = idx + offset;
    if (k < N) C[idx] = __ldg(&A[k]) + __ldg(&B[k]);
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

void sumArraysOnHost(float *A, float *B, float *C, const int N, const int offset) {
    for (int i = 0, k = offset; k < N; i++, k++) {
        C[i] = A[k] + B[k];
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
    return;
}