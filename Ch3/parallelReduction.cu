#include <stdio.h>
#include <cuda_runtime.h>

void initialData(int *ip, const int size);
int naiveReduce(int *data, int size);
int neighboredPairReduce(int *data, const int size);
int interleavedPairReduce(int *data, const int size);

__global__ void reduceNeighbored(int *g_idata, int * g_odata, const int n);
__global__ void reduceNeighboredLessDiv(int *g_idata, int * g_odata, const int n);
__global__ void reduceInterleaved(int *g_idata, int * g_odata, const int n);

__global__ void reduceUnrolling2(int *g_idata, int * g_odata, const int n);
__global__ void reduceUnrolling4(int *g_idata, int * g_odata, const int n);
__global__ void reduceUnrolling8(int *g_idata, int * g_odata, const int n);

__global__ void reduceUnrollWraps8(int *g_idata, int * g_odata, const int n);
__global__ void reduceCompleteUnrollWarps8(int *g_idata, int * g_odata, const int n);

template <int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int * g_odata, const int n);

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

    // 0. compute on CPU
    start = clock();
    // int cpuRes = naiveReduce(h_idata_cpy, evenSize);
    // int cpuRes = neighboredPairReduce(h_idata_cpy, evenSize);
    reductionSum = interleavedPairReduce(h_idata_cpy, evenSize);
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nCPU reduce:                   execution time %.2f ms, result %d\n", exeTime * 1e3, reductionSum);

    // allocate device memory
    int *d_idata, *d_odata;
    CHECK(cudaMalloc((int**)&d_idata, nBytes));
    CHECK(cudaMalloc((int**)&d_odata, grid.x * sizeof(int)));

    // just warm up
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());

    // 1. GPU - neighbored pair reduce
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
    printf("GPU neighbored pair reduce:   execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

    // 2. GPU - neighbored pair reduce 2
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    reduceNeighboredLessDiv<<<grid, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) reductionSum += h_odata[i];
    printf("GPU neighbored pair reduce 2: execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

    // 3. GPU - interleaved pair reduce
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) reductionSum += h_odata[i];
    printf("GPU interleaved pair reduce:  execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

    // 4. GPU - interleaved pair reduce; x2 unrolling
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x / 2; i++) reductionSum += h_odata[i];
    printf("GPU x2 unrolling:             execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

    // 5. GPU - interleaved pair reduce; x4 unrolling
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x / 4; i++) reductionSum += h_odata[i];
    printf("GPU x4 unrolling:             execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

    // 6. GPU - interleaved pair reduce; x8 unrolling
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    reduceUnrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x / 8; i++) reductionSum += h_odata[i];
    printf("GPU x8 unrolling:             execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

    // 7. GPU - unrolling wraps
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    reduceUnrollWraps8<<<grid.x / 8, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x / 8; i++) reductionSum += h_odata[i];
    printf("GPU unrolling wraps:          execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

    // 8. GPU - complete unrolling 
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, evenSize);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x / 8; i++) reductionSum += h_odata[i];
    printf("GPU complete unrolling:       execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

    // 9. GPU - template function 
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    start = clock();
    // CUDA part
    switch (blockSize) {
        case 1024:
            reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata, evenSize);
            break;
        case 512:
            reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata, evenSize);
            break;
        case 256:
            reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata, evenSize);
            break;
        case 128:
            reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata, evenSize);
            break;
        case 64:
            reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, evenSize);
            break;
    }
    CHECK(cudaDeviceSynchronize());
    end = clock();
    exeTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Host part
    reductionSum = 0;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x / 8; i++) reductionSum += h_odata[i];
    printf("GPU template function:        execution time %.4f ms, result %d\n", exeTime * 1e3, reductionSum);

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


__global__ void reduceNeighboredLessDiv(int *g_idata, int * g_odata, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
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
    if (idx >= n) return;
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


__global__ void reduceUnrolling2(int *g_idata, int * g_odata, const int n) {
    const int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (idx >= n) return;
    const int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2 data blocks
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads(); 
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrolling4(int *g_idata, int * g_odata, const int n) {
    const int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    if (idx >= n) return;
    const int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx + 3 * blockDim.x < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads(); 
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrolling8(int *g_idata, int * g_odata, const int n) {
    const int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    if (idx >= n) return;
    const int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads(); 
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrollWraps8(int *g_idata, int * g_odata, const int n) {
    const int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    if (idx >= n) return;
    const int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;

        // int a1 = idata[tid];
        // int a2 = idata[tid + blockDim.x];
        // int a3 = idata[tid + 2 * blockDim.x];
        // int a4 = idata[tid + 3 * blockDim.x];
        // int a5 = idata[tid + 4 * blockDim.x];
        // int a6 = idata[tid + 5 * blockDim.x];
        // int a7 = idata[tid + 6 * blockDim.x];
        // int a8 = idata[tid + 7 * blockDim.x];
        // idata[tid] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads(); 
    }

    // unrolling wrap
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8(int *g_idata, int * g_odata, const int n) {
    const int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    if (idx >= n) return;
    const int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    // unrolling wrap
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

template <int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int * g_odata, const int n) {
    const int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    if (idx >= n) return;
    const int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    // unrolling wrap
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
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