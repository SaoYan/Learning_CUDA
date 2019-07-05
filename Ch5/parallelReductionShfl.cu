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

#define BLOCKSIZE 256
#define MASK 0xffffffff

void initialData(int *ip, const int size);
int naiveReduce(int *data, int size);
int neighboredPairReduce(int *data, const int size);
int interleavedPairReduce(int *data, const int size);

__global__ void reduceSharedMem(int *g_idata, int * g_odata, const int n);
__global__ void reduceShfl (int *g_idata, int *g_odata, const int n);

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
    dim3 block(BLOCKSIZE);
    dim3 grid((evenSize + block.x - 1)/ block.x);
    printf("Grid dimension %d Block dimensiton %d\n", grid.x, block.x);

    // allocate host memory
    int *h_idata, *h_odata, *h_idata_cpy;
    h_idata = (int *) malloc(nBytes);
    h_odata = (int *) malloc(grid.x * sizeof(int));
    h_idata_cpy = (int *) malloc(nBytes);
    memset(h_idata, 0, nBytes);
    initialData(h_idata, size);
    memcpy(h_idata_cpy, h_idata, nBytes);

    // 0. compute on CPU
    start = clock();
    // reductionSum = naiveReduce(h_idata_cpy, size);
    // reductionSum = neighboredPairReduce(h_idata_cpy, evenSize);
    reductionSum = interleavedPairReduce(h_idata_cpy, evenSize);
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nCPU reduce: execution time %.4f ms, result %d\n\n", exeTime * 1e3, reductionSum);

    // allocate device memory
    int *d_idata, *d_odata;
    CHECK(cudaMalloc((int**)&d_idata, nBytes));
    CHECK(cudaMalloc((int**)&d_odata, grid.x * sizeof(int)));

    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_odata, 0, grid.x * sizeof(int)));
    memset(h_odata, 0, grid.x * sizeof(int));
    start = clock();
    // CUDA part
    reduceShfl<<<grid.x, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) reductionSum += h_odata[i];
    printf("GPU warp shuffle: execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);
    CHECK(cudaGetLastError());

    // free host mem
    free(h_idata);
    free(h_odata);
    free(h_idata_cpy);

    // free device mem
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));
    
    // clean up all resources
    CHECK(cudaDeviceReset());
    return 0;
}

/**********CUDA kernels**********/

__global__ void reduceSharedMem(int *g_idata, int * g_odata, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // shared memory
    __shared__ int smem[BLOCKSIZE];
    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int *vmem = smem;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__inline__ __device__ int warpReduce(int localSum) {
    localSum += __shfl_xor_sync(MASK, localSum, 16);
    localSum += __shfl_xor_sync(MASK, localSum, 8);
    localSum += __shfl_xor_sync(MASK, localSum, 4);
    localSum += __shfl_xor_sync(MASK, localSum, 2);
    localSum += __shfl_xor_sync(MASK, localSum, 1);
    return localSum;
}

__global__ void reduceShfl (int *g_idata, int *g_odata, const int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // shared memory for each warp sum
    __shared__ int smem[BLOCKSIZE/32];

    // calculate lane index and warp index
    int laneIdx = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;

    // blcok-wide warp reduce
    int localSum = warpReduce(g_idata[idx]);

    // save warp sum to shared memory
    if (laneIdx == 0) smem[warpIdx] = localSum;
    __syncthreads();

    // last warp reduce
    if (threadIdx.x < warpSize) 
        localSum = (threadIdx.x < BLOCKSIZE/warpSize) ? smem[laneIdx] : 0;
    if (warpIdx == 0) localSum = warpReduce(localSum);

    // write result for this block to global mem
    if (threadIdx.x == 0) g_odata[blockIdx.x] = localSum;
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