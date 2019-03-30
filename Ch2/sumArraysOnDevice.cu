#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

void initialData(float *ip, int size);
void sumArraysOnHost(float *A, float *B, float *C, const int N);


__global__ void sumArraysOnDevice(float *A, float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}


int main(int argc, char **argv) {
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);

    // allocate host memory
    float *h_A, *h_B, *h_C;
    h_A = (float *) malloc(nBytes);
    h_B = (float *) malloc(nBytes);
    h_C = (float *) malloc(nBytes);

    // initial data (in CPU mem)
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    // allocate device mem
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, nBytes);
    cudaMalloc(&d_B, nBytes);
    cudaMalloc(&d_C, nBytes);

    // copy data from CPU to GPU
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // launch CUDA kernel
    dim3 block(nElem);
    dim3 grid((nElem+block.x-1)/block.x);
    kernel<<<grid, block>>>(d_A, d_B, d_C);
    
    // copy data from GPU back to CPU
    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

    // free host mem
    free(h_A);
    free(h_B);
    free(h_C);

    // free device mem
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}

void initialData(float *ip, int size) {
    // generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xFF) / 10.f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}