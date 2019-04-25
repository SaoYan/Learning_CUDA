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

#define BLOCKX 32
#define BLOCKY 32
#define PADDING 1

void initialMatrix(float *ip, int nx, int ny);
void checkResult(float *hostRef, float *gpuRef, const int nx, const int ny);

void transposeHostRowCol(float *in, float *out, const int nx, const int ny);
void transposeHostColRow(float *in, float *out, const int nx, const int ny);

__global__ void transposeGlobalRowCol(float *in, float *out, int nx, int ny);
__global__ void transposeGlobalColRow(float *in, float *out, int nx, int ny);

__global__ void transposeShared(float *in, float *out, int nx, int ny);
__global__ void transposeSharedPad(float *in, float *out, int nx, int ny);
__global__ void transposeSharedPadUnroll(float *in, float *out, int nx, int ny);

int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // configuration
    int nx = 1<<11, ny = 1<<11;
    if (argc > 1) nx = atoi(argv[1]);
    if (argc > 2) ny = atoi(argv[2]);
    dim3 block(BLOCKX, BLOCKY);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    printf("Matrix size (%d, %d)\n", nx, ny);
    printf("Kernel execution config <<<(%d, %d), (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

    size_t nBytes = nx * ny * sizeof(float);
    clock_t start, end;
    double time;

    // allocate host memory
    float *h_in      = (float *)malloc(nBytes);
    float *h_out     = (float *)malloc(nBytes);
    float *h_out_gpu = (float *)malloc(nBytes);
    initialMatrix(h_in, nx, ny);

    // compute on CPU
    start = clock();
    transposeHostRowCol(h_in, h_out, nx, ny);
    // transposeHostColRow(h_in, h_out, nx, ny);
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("transpose on CPU: %f ms\n\n", time * 1000.0);

    // allocate device memory
    float *d_in, *d_out;
    CHECK(cudaMalloc((float**)&d_in, nBytes));
    CHECK(cudaMalloc((float**)&d_out, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

    printf("Baseline: only using global memory:\n");

    // baseline; read by row, store by col
    memset(h_out_gpu, 0.0, nBytes);
    CHECK(cudaMemset(d_out, 0.0, nBytes));
    start = clock();
    transposeGlobalRowCol<<<grid, block>>>(d_in, d_out, nx, ny);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("read by row, store by column: %f ms\n", time * 1000.0);
    CHECK(cudaMemcpy(h_out_gpu, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, h_out_gpu, nx, ny);

    // baseline; read by col, store by row
    memset(h_out_gpu, 0.0, nBytes);
    CHECK(cudaMemset(d_out, 0.0, nBytes));
    start = clock();
    transposeGlobalColRow<<<grid, block>>>(d_in, d_out, nx, ny);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("read by column, store by row: %f ms\n\n", time * 1000.0);
    CHECK(cudaMemcpy(h_out_gpu, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, h_out_gpu, nx, ny);

    printf("Using shared memory:\n");

    // shared memory
    memset(h_out_gpu, 0.0, nBytes);
    CHECK(cudaMemset(d_out, 0.0, nBytes));
    start = clock();
    transposeShared<<<grid, block>>>(d_in, d_out, nx, ny);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Basic implementstion: %f ms\n", time * 1000.0);
    CHECK(cudaMemcpy(h_out_gpu, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, h_out_gpu, nx, ny);

    // shared memory with padding
    memset(h_out_gpu, 0.0, nBytes);
    CHECK(cudaMemset(d_out, 0.0, nBytes));
    start = clock();
    transposeSharedPad<<<grid, block>>>(d_in, d_out, nx, ny);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Memory padding:       %f ms\n", time * 1000.0);
    CHECK(cudaMemcpy(h_out_gpu, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, h_out_gpu, nx, ny);

    // shared memory with padding and unrolling
    dim3 grid2((nx + block.x - 1) / block.x / 2, (ny + block.y - 1) / block.y);
    memset(h_out_gpu, 0.0, nBytes);
    CHECK(cudaMemset(d_out, 0.0, nBytes));
    start = clock();
    transposeSharedPadUnroll<<<grid2, block>>>(d_in, d_out, nx, ny);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Padding + Unrolling:  %f ms\n", time * 1000.0);
    CHECK(cudaMemcpy(h_out_gpu, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, h_out_gpu, nx, ny);

    // free memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    free(h_out_gpu);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}


/**********CUDA kernels**********/

__global__ void transposeGlobalRowCol(float *in, float *out, int nx, int ny) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void transposeGlobalColRow(float *in, float *out, int nx, int ny) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

__global__ void transposeShared(float *in, float *out, int nx, int ny) {
    // shared memory
    __shared__ float smem[BLOCKY][BLOCKX];

    // global memory index for original matrix
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // transposed index in shared memory
    int irow = (threadIdx.y * blockDim.x + threadIdx.x) % blockDim.y;
    int icol = (threadIdx.y * blockDim.x + threadIdx.x) / blockDim.y;

    // global memory index for transposed matrix
    int ox = blockDim.y * blockIdx.y + irow;
    int oy = blockDim.x * blockIdx.x + icol;

    if (ix < nx && iy < ny) {
        smem[threadIdx.y][threadIdx.x] = in[iy * nx + ix];
        __syncthreads();
        out[oy * ny + ox] = smem[irow][icol];
    }
}

__global__ void transposeSharedPad(float *in, float *out, int nx, int ny) {
    // shared memory
    __shared__ float smem[BLOCKY][BLOCKX+PADDING];

    // global memory index for original matrix
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // transposed index in shared memory
    int irow = (threadIdx.y * blockDim.x + threadIdx.x) % blockDim.y;
    int icol = (threadIdx.y * blockDim.x + threadIdx.x) / blockDim.y;

    // global memory index for transposed matrix
    int ox = blockDim.y * blockIdx.y + irow;
    int oy = blockDim.x * blockIdx.x + icol;

    if (ix < nx && iy < ny) {
        smem[threadIdx.y][threadIdx.x] = in[iy * nx + ix];
        __syncthreads();
        out[oy * ny + ox] = smem[irow][icol];
    }
}

__global__ void transposeSharedPadUnroll(float *in, float *out, int nx, int ny) {
    // shared memory
    __shared__ float smem[BLOCKY][BLOCKX*2+PADDING];

    // global memory index for original matrix
    int ix = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // transposed index in shared memory
    int irow = (threadIdx.y * blockDim.x + threadIdx.x) % blockDim.y;
    int icol = (threadIdx.y * blockDim.x + threadIdx.x) / blockDim.y;

    // global memory index for transposed matrix
    int ox = blockDim.y * blockIdx.y + irow;
    int oy = blockDim.x * blockIdx.x * 2 + icol;

    if (ix + blockDim.x < nx && iy < ny) {
        smem[threadIdx.y][threadIdx.x] = in[iy * nx + ix];
        smem[threadIdx.y][threadIdx.x+blockDim.x] = in[iy * nx + ix + blockDim.x];
        __syncthreads();
        out[oy * ny + ox] = smem[irow][icol];
        out[(oy + blockDim.x) * ny + ox] = smem[irow][icol+blockDim.x];
    }
}

/**********host functions**********/

// read by rows (coalesced access); store by columns (strided access)
void transposeHostRowCol(float *in, float *out, const int nx, const int ny) { 
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) { 
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

// read by columns (strided access); store by rows (coalesced access)
void transposeHostColRow(float *in, float *out, const int nx, const int ny) { 
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) { 
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

void initialMatrix(float *ip, int nx, int ny) {
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) { 
            ip[ix * ny + iy] = (float)(rand() & 0xFF) / 100.0f;
        }
    }
}

void checkResult(float *hostRef, float *gpuRef, const int nx, const int ny) {
    double epsilon = 1.0E-8;
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) { 
            int idx = ix * ny + iy;
            if (abs(hostRef[idx] - gpuRef[idx]) > epsilon) {
                printf("different on (x %d, y %d): host %f gpu %f\n", 
                    ix, iy, hostRef[idx], gpuRef[idx]);
                return;
            }
        }
    }
}