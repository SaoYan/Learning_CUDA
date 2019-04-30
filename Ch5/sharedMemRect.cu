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

#define BDIMX 32
#define BDIMY 16
#define PADDING 2 // with 1-padding, there will still be 2-way conflict (analyze yourself~)

void printData(const char *msg, int *data, const int size);

__global__ void writeRowReadRow(int *out);
__global__ void writeColReadCol(int *out);
__global__ void writeColReadColT(int *out);
__global__ void writeRowReadCol(int *out);
__global__ void writeColReadRow(int *out);

__global__ void writeRowReadRowDynamic(int *out);
__global__ void writeColReadColDynamic(int *out);

__global__ void writeColReadColPad(int *out);
__global__ void writeColReadColDynamicPad(int *out);

int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    cudaSharedMemConfig pConfig;
    CHECK(cudaDeviceGetSharedMemConfig(&pConfig));
    printf("Shared memory bank width: %s\n", pConfig == 1 ? "4-Byte" : "8-Byte");

    // allocate memory
    size_t nBytes = BDIMX * BDIMX * sizeof(int);
    int *d_out;
    CHECK(cudaMalloc((int**)&d_out, nBytes));
    int *h_out  = (int*)malloc(nBytes);

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);
    bool ifprint = 0; // print result or not
    if (argc > 1) ifprint = atoi(argv[1]);

    CHECK(cudaMemset(d_out, 0, nBytes));
    memset(h_out, 0, nBytes);
    writeRowReadRow<<<grid, block>>>(d_out);
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (ifprint) printData("write row read row:\n", h_out, BDIMX * BDIMY);
    CHECK(cudaGetLastError());

    CHECK(cudaMemset(d_out, 0, nBytes));
    memset(h_out, 0, nBytes);
    writeColReadCol<<<grid, block>>>(d_out);
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (ifprint) printData("write col read col:\n", h_out, BDIMX * BDIMY);
    CHECK(cudaGetLastError());

    CHECK(cudaMemset(d_out, 0, nBytes));
    memset(h_out, 0, nBytes);
    writeColReadColT<<<grid, block>>>(d_out);
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (ifprint) printData("write col read col; transpose:\n", h_out, BDIMX * BDIMY);
    CHECK(cudaGetLastError());

    CHECK(cudaMemset(d_out, 0, nBytes));
    memset(h_out, 0, nBytes);
    writeRowReadCol<<<grid, block>>>(d_out);
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (ifprint) printData("write row read col:\n", h_out, BDIMX * BDIMY);
    CHECK(cudaGetLastError());

    CHECK(cudaMemset(d_out, 0, nBytes));
    memset(h_out, 0, nBytes);
    writeColReadRow<<<grid, block>>>(d_out);
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (ifprint) printData("write col read row:\n", h_out, BDIMX * BDIMY);
    CHECK(cudaGetLastError());

    CHECK(cudaMemset(d_out, 0, nBytes));
    memset(h_out, 0, nBytes);
    writeRowReadRowDynamic<<<grid, block, nBytes>>>(d_out);
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (ifprint) printData("write row read row; dynamic:\n", h_out, BDIMX * BDIMY);
    CHECK(cudaGetLastError());

    CHECK(cudaMemset(d_out, 0, nBytes));
    memset(h_out, 0, nBytes);
    writeColReadColDynamic<<<grid, block, nBytes>>>(d_out);
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (ifprint) printData("write col read col; dynamic:\n", h_out, BDIMX * BDIMY);
    CHECK(cudaGetLastError());

    CHECK(cudaMemset(d_out, 0, nBytes));
    memset(h_out, 0, nBytes);
    writeColReadColPad<<<grid, block, nBytes>>>(d_out);
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (ifprint) printData("write col read col; padding:\n", h_out, BDIMX * BDIMY);
    CHECK(cudaGetLastError());

    CHECK(cudaMemset(d_out, 0, nBytes));
    memset(h_out, 0, nBytes);
    writeColReadColDynamicPad<<<grid, block, nBytes + PADDING*BDIMY*sizeof(int)>>>(d_out);
    CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (ifprint) printData("write col read col; dynamic + padding:\n", h_out, BDIMX * BDIMY);
    CHECK(cudaGetLastError());

    CHECK(cudaFree(d_out));
    CHECK(cudaDeviceReset());
    return 0;
}

/**********CUDA kernels**********/

__global__ void writeRowReadRow(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    // assuming only one block
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    // global memory store operation
    out[idx] = tile[threadIdx.y][threadIdx.x] ;
}

__global__ void writeColReadCol(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    // assuming only one block
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    int irow = idx % BDIMY;
    int icol = idx / BDIMY;
    tile[irow][icol] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    // global memory store operation
    out[idx] = tile[irow][icol];
}

__global__ void writeColReadColT(int *out) {
    // static shared memory; transposed
    __shared__ int tile[BDIMX][BDIMY];

    // mapping from thread index to global memory index
    // assuming only one block
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    // global memory store operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void writeRowReadCol(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    // assuming only one block
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    // global memory store operation
    int irow = idx % BDIMY;
    int icol = idx / BDIMY;
    out[idx] = tile[irow][icol];
}

__global__ void writeColReadRow(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    // assuming only one block
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    int irow = idx % BDIMY;
    int icol = idx / BDIMY;
    tile[irow][icol] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    // global memory store operation
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void writeRowReadRowDynamic(int *out) {
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    // assuming only one block
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[idx] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    // global memory store operation
    out[idx] = tile[idx];
}

__global__ void writeColReadColDynamic(int *out) {
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    // assuming only one block
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    int irow = idx % BDIMY;
    int icol = idx / BDIMY;
    int colIdx = irow * BDIMX + icol; // col-based index
    tile[colIdx] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    // global memory store operation
    out[idx] = tile[colIdx];
}

__global__ void writeColReadColPad(int *out) {
    // static shared memory with memory padding
    __shared__ int tile[BDIMY][BDIMX + PADDING];

    // mapping from thread index to global memory index
    // assuming only one block
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    int irow = idx % BDIMY;
    int icol = idx / BDIMY;
    tile[irow][icol] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    // global memory store operation
    out[idx] = tile[irow][icol];
}

__global__ void writeColReadColDynamicPad(int *out) {
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    // assuming only one block
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    int irow = idx % BDIMY;
    int icol = idx / BDIMY;
    int colIdx = irow * (BDIMX + PADDING) + icol; // col-based index
    tile[colIdx] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    // global memory store operation
    out[idx] = tile[colIdx];
}

/**********host functions**********/

void printData(const char *msg, int *data, const int size) {
    printf("%s: ", msg);

    for (int i = 0; i < size; i++) {
        printf("%d ", data[i]);
        fflush(stdout);
    }

    printf("\n\n");
    return;
}