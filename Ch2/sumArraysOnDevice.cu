#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

void initialData(float *ip, int rows, int cols);
void sumArraysOnHost(float *A, float *B, float *C, const int rows, const int cols);
void verifyResult(float *hostRes, float *deviceRes, int rows, int cols);

__global__ void sumArraysOnDevice(float *A, float *B, float *C);

int main(int argc, char **argv) {
    // On current GPUs, a thread block may contain up to 1024 threads.
    int rows = 409600, cols = 512;
    size_t nBytes = rows * cols * sizeof(float);
    clock_t start, end;
    cudaError_t error;

    // set up device 
    int dev = 0; 
    cudaSetDevice(dev);

    // allocate host memory
    float *h_A, *h_B, *h_C, *h_C_gpu;
    h_A = (float *) malloc(nBytes);
    h_B = (float *) malloc(nBytes);
    h_C = (float *) malloc(nBytes);
    h_C_gpu = (float *) malloc(nBytes);

    // initial data (in CPU mem)
    initialData(h_A, rows, cols);
    initialData(h_B, rows, cols);
    memset(h_C, 0, nBytes);
    memset(h_C_gpu, 0, nBytes);

    // compute on CPU
    start = clock();
    sumArraysOnHost(h_A, h_B, h_C, rows, cols);
    end = clock();
    double cpuTime = ((double) (end - start)) / CLOCKS_PER_SEC;

    // allocate device mem
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    error = cudaMalloc((float**)&d_C, nBytes);
    printf("Allocating GPU mem: %s\n", cudaGetErrorString(error));

    // copy data from CPU to GPU
    start = clock();
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    error = cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    end = clock();
    double copyTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Copying data CPU --> GPU: %s\n", cudaGetErrorString(error));

    // launch CUDA kernel
    dim3 block(cols);
    dim3 grid((rows*cols+block.x-1)/block.x);
    start = clock();
    sumArraysOnDevice<<<grid, block>>>(d_A, d_B, d_C);
    end = clock();
    double gpuTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    // copy data from GPU back to CPU
    error = cudaMemcpy(h_C_gpu, d_C, nBytes, cudaMemcpyDeviceToHost);
    printf("Copyint data GPU --> CPU: %s\n", cudaGetErrorString(error));

    // verify
    verifyResult(h_C, h_C_gpu, rows, cols);
    printf("It takes %.2f sec to execute on CPU\n", cpuTime);
    printf("It takes %.2f sec to copy data from CPU to GPU\n", copyTime);
    printf("It takes %.2f sec to execute on GPU\n", gpuTime);

    // free host mem
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_gpu);

    // free device mem
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // clean up all resources
    cudaDeviceReset();

    return 0;
}

/**********CUDA kernels**********/

__global__ void sumArraysOnDevice(float *A, float *B, float *C) {
    // 1D grid of 1D block
    // compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    C[idx] = A[idx] + B[idx];
}

/**********host functions**********/

void initialData(float *ip, int rows, int cols) {
    // generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            ip[idx] = (float) (rand() & 0xFF) / 10.f;
        }
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            C[idx] = A[idx] + B[idx];
        }
    }
}

void verifyResult(float *hostRes, float *deviceRes, int rows, int cols) {
    double eps = 1e-8;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            if (abs(hostRes[idx] - deviceRes[idx] > eps)) {
                printf("Arrays do not match! Check your kernel code!\n");
                printf("host %5.2f gpu %5.2f at (%d, %d)\n", hostRes[idx], deviceRes[idx], i, j);
                return;
            }
        }
    }
    printf("Arrays match! Congrats, your kernel code works well!\n");
    return;
}