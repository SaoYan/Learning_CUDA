#include <stdio.h>
#include <cuda_runtime.h>

// reference
// https://devblogs.nvidia.com/maximizing-unified-memory-performance-cuda/

#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}                                                                            \

#define PAGE_STRIDE 65536 // page size: 64K -> 65536 bytes

template <typename data_type>
__global__ void stream_thread(data_type *input, data_type *output, const int n);

template <typename data_type>
__global__ void stream_warp(data_type *input, data_type *output, const int n);

void initialData(float *ip, const int n);
void verifyResult(float *result, float *reference, const int n);

int main(int argc, char **argv) {
    int n = 1<<20;
    size_t nBytes = n * sizeof(float);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // allocate unified memory
    float *in, *out;
    CHECK(cudaMallocManaged((float**)&in, nBytes));
    CHECK(cudaMallocManaged((float**)&out, nBytes));
    CHECK(cudaMemPrefetchAsync(in,  nBytes, cudaCpuDeviceId, 0));
    initialData(in, n);

    // launch kernels
    int caseNo = 1;
    if (argc > 1) caseNo = atoi(argv[1]);
    int blockSize = 256;
    if (argc > 2) blockSize = atoi(argv[2]);
    dim3 block(blockSize);
    dim3 grid((n + block.x - 1) / block.x);
    int pages = (nBytes + PAGE_STRIDE - 1) / PAGE_STRIDE; // # pages
    switch (caseNo) {
        case 1: // no pre-fetching; normal kernel
            grid.x = (n + block.x - 1) / block.x;
            printf("<<< %d, %d >>>\n", grid.x, block.x);
            stream_thread<float> <<<grid, block>>> (in, out, n);
            break;
        case 2: // no pre-fetching; one warp per page
            grid.x = (pages * 32 + block.x - 1) / block.x; // # warps = # pages
            printf("<<< %d, %d >>>\n", grid.x, block.x);
            stream_warp<float> <<<grid, block>>> (in, out, n);
            break;
        case 3: // pre-fetching; normal kernel
            CHECK(cudaMemPrefetchAsync(in,  nBytes, dev, 0));
            CHECK(cudaMemPrefetchAsync(out, nBytes, dev, 0));
            grid.x = (n + block.x - 1) / block.x;
            printf("<<< %d, %d >>>\n", grid.x, block.x);
            stream_thread<float> <<<grid, block>>> (in, out, n);
            break;
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemPrefetchAsync(in,  nBytes, cudaCpuDeviceId, 0));
    CHECK(cudaMemPrefetchAsync(out, nBytes, cudaCpuDeviceId, 0));
    verifyResult(out, in, n);
    CHECK(cudaGetLastError());

    // free memory
    CHECK(cudaFree(in));
    CHECK(cudaFree(out));
    
    // clean up all resources
    CHECK(cudaDeviceReset());
    return 0;
}

/**********CUDA kernels**********/

template <typename data_type>
__global__ void stream_thread(data_type *input, data_type *output, const int n) { 
    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    if (tid < n) {
        output[tid] = input[tid];
    }
}

template <typename data_type>
__global__ void stream_warp(data_type *input, data_type *output, const int n) { 
    int laneId = threadIdx.x & 31; 
    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    int warpId = tid >> 5;
    size_t size = n * sizeof(data_type);
    int pages = (size + PAGE_STRIDE - 1) / PAGE_STRIDE; // how many pages
    if (warpId < pages) {
        #pragma unroll // tell compiler to specifically unroll a loop
        for(int rep = 0; rep < PAGE_STRIDE / sizeof(data_type) / 32; rep++) {
            int ind = warpId * PAGE_STRIDE / sizeof(data_type) + rep * 32 + laneId;
            if (ind < n)
                output[ind] = input[ind];
        }
    }
}

/**********host functions**********/

void initialData(float *ip, const int n) {
    for (int i = 0; i < n; i++) {
        ip[i] = (float) (rand() & 0xFF) / 10.f;
    }
}

void verifyResult(float *result, float *reference, const int n) {
    double eps = 1e-8;
    bool match = 1;
    for (int i = 0; i < n; i++) {
        if (abs(result[i] - reference[i]) > eps) {
            printf("Arrays do not match:\n");
            printf("result %5.2f reference %5.2f at array index %d\n", result[i], reference[i], i);
            match = 0;
            return;
        }
    }
    if (match) printf("Arrays match!\n");
    return;
}