#include <stdio.h>
#include <cuda_runtime.h>

void initialData(int *ip, const int size);
int cpuRecursiveReduce(int *data, int const size);
int naiveReduce(int *data, int size);

__global__ void reduceNeighbored(int *g_idata, int * g_odata, const int n);
__global__ void gpuRecursiveReduce (int *g_idata, int *g_odata, const int n);
__global__ void gpuRecursiveReduceNosync (int *g_idata, int *g_odata, const int n);
__global__ void gpuRecursiveReduce2 (int *g_idata, int *g_odata, const int stride, const int blkDim);

#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}                                                                            \

int main(int argc, char **argv) {
    // int size = 1 << 24, evenSize = size;
    int size = 2048 * 512, evenSize = size;
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
    reductionSum = cpuRecursiveReduce(h_idata_cpy, evenSize);
    // reductionSum = naiveReduce(h_idata_cpy, size);
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nCPU reduce:                     execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

    // allocate device memory
    int *d_idata, *d_odata;
    CHECK(cudaMalloc((int**)&d_idata, nBytes));
    CHECK(cudaMalloc((int**)&d_odata, grid.x * sizeof(int)));

    // just warm up
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());

    // 1. GPU - neighbored pair reduce; divergence
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) reductionSum += h_odata[i];
    printf("GPU neighbored pair reduce:     execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

    // 2. GPU - recursive
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    gpuRecursiveReduce<<<grid, block>>>(d_idata, d_odata, block.x);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) reductionSum += h_odata[i];
    printf("GPU recursive reduce:           execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

    // 3. GPU - recursive; no sync
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    gpuRecursiveReduceNosync<<<grid, block>>>(d_idata, d_odata, block.x);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) reductionSum += h_odata[i];
    printf("GPU recursive reduce no sync:   execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

    // 4. GPU - recursive pattern 2
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    gpuRecursiveReduce2<<<grid, block.x / 2>>>(d_idata, d_odata, block.x / 2, block.x);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) reductionSum += h_odata[i];
    printf("GPU recursive reduce pattern 2: execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

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

// Neighbored Pair Implementation with divergence
__global__ void reduceNeighbored(int *g_idata, int * g_odata, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
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

__global__ void gpuRecursiveReduce (int *g_idata, int *g_odata, const int n) {
    int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = g_odata + blockIdx.x;
    // int *odata = &g_odata[blockIdx.x];

    // stop condition
    if (n == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int stride = n >> 1;
    if(stride > 1 && tid < stride) {
        // in place reduction
        idata[tid] += idata[tid + stride];
    }
    __syncthreads(); // sync at block level

    // nested invocation to generate child grids
    if (tid == 0) {
        gpuRecursiveReduce<<<1, stride>>>(idata, odata, stride);
        // sync all child grids launched in this block
        cudaDeviceSynchronize();
    }
    __syncthreads(); // sync at block level again
}

__global__ void gpuRecursiveReduceNosync (int *g_idata, int *g_odata, const int n) {
    int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = g_odata + blockIdx.x;
    // int *odata = &g_odata[blockIdx.x];

    // stop condition
    if (n == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int stride = n >> 1;
    if(stride > 1 && tid < stride) {
        // in place reduction
        idata[tid] += idata[tid + stride];
        if (tid == 0) {
            gpuRecursiveReduceNosync<<<1, stride>>>(idata, odata, stride);
        }
    }
}

__global__ void gpuRecursiveReduce2 (int *g_idata, int *g_odata, const int stride, const int blkDim) {
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blkDim;

    // stop condition
    if (stride == 1 && threadIdx.x == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    // in place reduction
    idata[threadIdx.x] += idata[threadIdx.x + stride];

    // nested invocation to generate child grids
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        gpuRecursiveReduce2<<<gridDim.x, stride / 2>>>(g_idata, g_odata, stride / 2, blkDim);
    }
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

// Recursive Implementation of Interleaved Pair Approach
int cpuRecursiveReduce(int *data, int const size) {
    if (size == 1) return data[0];
    int const stride = size / 2;
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }
    return cpuRecursiveReduce(data, stride);
}

int naiveReduce(int *data, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}