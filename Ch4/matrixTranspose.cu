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

void initialMatrix(float *ip, int nx, int ny);
void checkResult(float *hostRef, float *gpuRef, const int nx, const int ny);

void transposeHostRowCol(float *out, float *in, const int nx, const int ny);
void transposeHostColRow(float *out, float *in, const int nx, const int ny);

__global__ void warmup(float *out, float *in, int nx, int ny);

__global__ void transposeRowCol(float *out, float *in, int nx, int ny);
__global__ void transposeColRow(float *out, float *in, int nx, int ny);

__global__ void transposeDiagonalRowCol(float *out, float *in, const int nx, const int ny);
__global__ void transposeDiagonalColRow(float *out, float *in, const int nx, const int ny);

// __global__ void transposeRowColUnroll4(float *out, float *in, const int nx, const int ny);
// __global__ void transposeColRowUnroll4(float *out, float *in, const int nx, const int ny);

int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // configuration
    int blockx = 16, blocky = 16;
    int nx = 1<<11, ny = 1<<11;
    if (argc > 1) blockx = atoi(argv[1]);
    if (argc > 2) blocky = atoi(argv[2]);
    if (argc > 3) nx = atoi(argv[3]);
    if (argc > 4) ny = atoi(argv[4]);
    dim3 block(blockx, blocky);
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
    printf("transpose on CPU:             %f ms\n", time * 1000.0);

    // allocate device memory
    float *d_in, *d_out;
    CHECK(cudaMalloc((float**)&d_in, nBytes));
    CHECK(cudaMalloc((float**)&d_out, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

    // 0. execute kernel - warm up
    memset(h_out_gpu, 0.0, nBytes);
    CHECK(cudaMemset(d_out, 0.0, nBytes));
    warmup<<<grid, block>>>(d_in, d_out, nx, ny);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!

    // 1. execute kernel - read by row, store by col
    memset(h_out_gpu, 0.0, nBytes);
    CHECK(cudaMemset(d_out, 0.0, nBytes));
    start = clock();
    transposeRowCol<<<grid, block>>>(d_in, d_out, nx, ny);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("[Cartesian] read by row, store by column: %f ms\n", time * 1000.0);
    CHECK(cudaMemcpy(h_out_gpu, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, h_out_gpu, nx, ny);

    // 2. execute kernel - read by col, store by row
    memset(h_out_gpu, 0.0, nBytes);
    CHECK(cudaMemset(d_out, 0.0, nBytes));
    start = clock();
    transposeColRow<<<grid, block>>>(d_in, d_out, nx, ny);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("[Cartesian] read by column, store by row: %f ms\n", time * 1000.0);
    CHECK(cudaMemcpy(h_out_gpu, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, h_out_gpu, nx, ny);

    // 3. execute kernel - read by col, store by row
    memset(h_out_gpu, 0.0, nBytes);
    CHECK(cudaMemset(d_out, 0.0, nBytes));
    start = clock();
    transposeDiagonalRowCol<<<grid, block>>>(d_in, d_out, nx, ny);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("[diagonal]  read by row, store by column: %f ms\n", time * 1000.0);
    CHECK(cudaMemcpy(h_out_gpu, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, h_out_gpu, nx, ny);

    // 4. execute kernel - read by col, store by row
    memset(h_out_gpu, 0.0, nBytes);
    CHECK(cudaMemset(d_out, 0.0, nBytes));
    start = clock();
    transposeDiagonalColRow<<<grid, block>>>(d_in, d_out, nx, ny);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("[diagonal]  read by column, store by row: %f ms\n", time * 1000.0);
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

__global__ void warmup(float *out, float *in, int nx, int ny) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void transposeRowCol(float *out, float *in, int nx, int ny) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void transposeColRow(float *out, float *in, int nx, int ny) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

__global__ void transposeDiagonalRowCol(float *out, float *in, const int nx, const int ny) {
    int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    int blk_y = blockIdx.x;

    int ix = blockDim.x * blk_x + threadIdx.x;
    int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void transposeDiagonalColRow(float *out, float *in, const int nx, const int ny) {
    int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    int blk_y = blockIdx.x;

    int ix = blockDim.x * blk_x + threadIdx.x;
    int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

// __global__ void transposeRowColUnroll4(float *out, float *in, const int nx, const int ny) {
//     int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
//     int iy = blockDim.y * blockIdx.y + threadIdx.y;

//     int ti = iy * nx + ix; // access in rows
//     int to = ix * ny + iy; // access in columns

//     if (ix + 3 * blockDim.x < nx && iy < ny) {
//         out[to]                       = in[ti];
//         out[to + ny * blockDim.x]     = in[ti + blockDim.x];
//         out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
//         out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
//     }
// }

// __global__ void transposeColRowUnroll4(float *out, float *in, const int nx, const int ny) {
//     int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
//     int iy = blockDim.y * blockIdx.y + threadIdx.y;

//     int ti = iy * nx + ix; // access in rows
//     int to = ix * ny + iy; // access in columns

//     if (ix + 3 * blockDim.x < nx && iy < ny) {
//         out[ti]                  = in[to];
//         out[ti +   blockDim.x]   = in[to +   blockDim.x * ny];
//         out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
//         out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
//     }
// }

/**********host functions**********/

// read by rows (coalesced access); store by columns (strided access)
void transposeHostRowCol(float *out, float *in, const int nx, const int ny) { 
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) { 
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

// read by columns (strided access); store by rows (coalesced access)
void transposeHostColRow(float *out, float *in, const int nx, const int ny) { 
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
                printf("different on (ix %d, iy %d): host %f gpu %f\n", 
                    ix, iy, hostRef[idx], gpuRef[idx]);
                break;
            }
        }
    }
}