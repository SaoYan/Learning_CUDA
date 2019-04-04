#include <stdio.h>
#include <cuda_runtime.h>

void initialData(int *ip, const int size);
int naiveReduce(int *data, int size);
int neighboredPairReduce(int *data, const int size);
int interleavedPairReduce(int *data, const int size);

__global__ void reduceNeighbored(int *g_idata, int * g_odata, const int n);
__global__ void reduceNeighboredLessDiv(int *g_idata, int * g_odata, const int n);
__global__ void reduceInterleaved(int *g_idata, int * g_odata, const int n);

#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}                                                                            \

int main(int argc, char **argv) {
    int size = 1<<24, evenSize = size;
    if (evenSize % 2 != 0) evenSize++; // should be even for pair-reducution to work
    printf("Vector size %d\n", size);

    size_t nBytes = evenSize * sizeof(int);
    clock_t start, end;
    double exeTime;
    int reductionSum;

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // grid and block configuration
    int blockSize = 512;
    if (argc > 1) { // support input arg from terminal
        blockSize = atoi(argv[1]);
    }
    dim3 block(blockSize);
    dim3 grid((evenSize + block.x - 1)/ block.x);
    printf("Grid dimension %d Block dimensiton %d\n",grid.x, block.x);

    // allocate host memory
    int *h_idata, *h_odata, *h_idata_cpy;
    h_idata = (int *) malloc(nBytes);
    h_odata = (int *) malloc(grid.x * sizeof(int));
    h_idata_cpy = (int *) malloc(nBytes);
    memset(h_idata, 0, nBytes);
    initialData(h_idata, size);
    memcpy(h_idata_cpy, h_idata, nBytes);

    // compute on CPU
    start = clock();
    // int cpuRes = naiveReduce(h_idata_cpy, evenSize);
    // int cpuRes = neighboredPairReduce(h_idata_cpy, evenSize);
    reductionSum = interleavedPairReduce(h_idata_cpy, evenSize);
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nCPU reduce: execution time %.2f ms, result %d\n", exeTime * 1e3, reductionSum);

    // allocate device memory
    int *d_idata, *d_odata;
    CHECK(cudaMalloc((int**)&d_idata, nBytes));
    CHECK(cudaMalloc((int**)&d_odata, grid.x * sizeof(int)));

    // GPU - neighbored pair reduce
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) reductionSum += h_odata[i];
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("GPU neighbored pair reduce: execution time %.2f ms, result %d\n", exeTime * 1e3, reductionSum);

    // GPU - neighbored pair reduce 2
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    reduceNeighboredLessDiv<<<grid, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) reductionSum += h_odata[i];
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("GPU neighbored pair reduce 2: execution time %.2f ms, result %d\n", exeTime * 1e3, reductionSum);

    // GPU - interleaved pair reduce
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) reductionSum += h_odata[i];
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("GPU interleaved pair reduce: execution time %.2f ms, result %d\n", exeTime * 1e3, reductionSum);

    // free host mem
    free(h_idata);
    free(h_odata);
    free(h_idata_cpy);

    // free device mem
    cudaFree(d_idata);
    cudaFree(d_odata);
    
    // clean up all resources
    cudaDeviceReset();
    return 0;
}

/**********CUDA kernels**********/

__global__ void reduceNeighbored(int *g_idata, int * g_odata, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n) return;
    const int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (stride * 2) == 0) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLessDiv(int *g_idata, int * g_odata, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n) return;
    const int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int i = tid * stride * 2;
        if (i < blockDim.x) {
            idata[i] += idata[i + stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int *g_idata, int * g_odata, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n) return;
    const int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads(); 
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

/**********host functions**********/

void initialData(int *ip, const int size) {
    // generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (int)( rand() & 0xFF );
    }
}

int naiveReduce(int *data, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

int neighboredPairReduce(int *data, const int size) {
    for (int stride = 1; stride <= size / 2; stride *= 2) {
        for (int i = 0; i < size; i += stride * 2) {
            data[i] += data[i + stride];
        }
    }
    return data[0];
}

int interleavedPairReduce(int *data, const int size) {
    if (size == 1) return data[0];
    const int stride = size / 2;
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }
    return interleavedPairReduce(data, stride);
}