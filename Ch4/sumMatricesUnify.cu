#include <stdio.h>
#include <cuda_runtime.h>

void initialData(float *ip, const int nx, const int ny);
void sumArraysOnHost(float *A, float *B, float *C, const int nx, const int ny);
void verifyResult(float *hostRes, float *deviceRes, const int nx, const int ny);

__global__ void sumArraysOnDevice(float *A, float *B, float *C, const int nx, const int ny);

#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}                                                                            \

int main(int argc, char **argv) {
    int nx = 1<<14, ny = 1<<14;
    size_t nBytes = nx * ny * sizeof(float);
    clock_t start, end;
    printf("Matrix size (%d, %d)\n", nx, ny);

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

    // initialize data
    initialData(A, nx, ny);
    initialData(B, nx, ny);
    memset(C, 0, nBytes);
    memset(C_gpu, 0, nBytes);

    // compute on CPU
    start = clock();
    sumArraysOnHost(A, B, C, nx, ny);
    end = clock();
    double cpuTime = ((double) (end - start)) / CLOCKS_PER_SEC;

    // configration
    int dimx = 16, dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    printf("Grid dimension (%d, %d) Block dimensiton (%d, %d)\n",grid.x, grid.y, block.x, block.y);

    // launch CUDA kernel
    memset(C_gpu, 0, nBytes);
    start = clock();
    sumArraysOnDevice<<<grid, block>>>(A, B, C_gpu, nx, ny);
    CHECK(cudaDeviceSynchronize()); // synchronization is necessary when using unified memory!
    end = clock();
    double gpuTime = ((double) (end - start)) / CLOCKS_PER_SEC;

    // verify
    verifyResult(C, C_gpu, nx, ny);
    CHECK(cudaGetLastError());
    printf("It takes %.4f sec to execute on CPU\n", cpuTime);
    printf("It takes %.4f sec to execute on GPU\n", gpuTime);

    // free memory
    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(C));
    CHECK(cudaFree(C_gpu));
    
    // clean up all resources
    CHECK(cudaDeviceReset());
    return 0;
}

/**********CUDA kernels**********/

__global__ void sumArraysOnDevice(float *A, float *B, float *C, const int nx, const int ny) {
    // Thread and block index --> Coordinate in the matrix
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // Coordinate in the matrix --> Offset in linear global memory  
    if (ix < nx && iy < ny)  {
        int idx = iy * nx + ix;
        C[idx] = A[idx] + B[idx];
    }
}

/**********host functions**********/

void initialData(float *ip, const int nx, const int ny) {
    // generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            int idx = iy * nx + ix;
            ip[idx] = (float) (rand() & 0xFF) / 10.f;
        }
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int nx, const int ny) {
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}

void verifyResult(float *hostRes, float *deviceRes, const int nx, const int ny) {
    double eps = 1e-8;
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            int idx = iy * nx + ix;
            if (abs(hostRes[idx] - deviceRes[idx]) > eps) {
                printf("Arrays do not match! Check your kernel code!\n");
                printf("host %5.2f gpu %5.2f at (%d, %d)\n", hostRes[idx], deviceRes[idx], nx, ny);
                return;
            }
        }
    }
    printf("Arrays match! Congrats, your kernel code works well!\n");
    return;
}