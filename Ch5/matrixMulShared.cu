#include <stdio.h>
#include <cuda_runtime.h>

// matrix multiplication
// matrix dimensions are assumed to be multiples of BLOCKSIZE

#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}                                                                            \

#define BLOCKSIZE 16

typedef struct {
    int width;
    int height;
    int stride;
    int* array;
} Matrix;

Matrix initialMatrixHost(int width, int height);
Matrix initialMatrixDevice(Matrix mHost, bool copy);
void checkResult(Matrix hostRef, Matrix gpuRef);

void matrixMulHost(Matrix A, Matrix B, Matrix C);

__global__ void matrixMulGlobal(Matrix A, Matrix B, Matrix C);
__global__ void matrixMulShared(Matrix A, Matrix B, Matrix C);

int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // configuration
    int widthA = 256, heightA = 1024, widthB = 1024, heightB = 512;
    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((widthB + block.x - 1) / block.x, (heightA + block.y - 1) / block.y);
    printf("Matrix size (height, width): A (%d, %d) B (%d, %d)\n", heightA, widthA, heightB, widthB);
    printf("Kernel execution config <<<(%d, %d), (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

    clock_t start, end;
    double time;

    // allocate host memory
    Matrix hA = initialMatrixHost(widthA, heightA);
    Matrix hB = initialMatrixHost(widthB, heightB);
    Matrix hC = initialMatrixHost(widthB, heightA);
    Matrix hC_gpu = initialMatrixHost(widthB, heightA);

    // compute on CPU
    start = clock();
    matrixMulHost(hA, hB, hC);
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nmatrix multiplication on CPU\t: %f ms\n", time * 1000.0);

    // allocate device memory
    Matrix dA = initialMatrixDevice(hA, 1);
    Matrix dB = initialMatrixDevice(hB, 1);
    Matrix dC = initialMatrixDevice(hC, 0);

    // compute on GPU; no shared memory
    int nBytes = dC.width * dC.height * sizeof(int);
    memset(hC_gpu.array, 0.0, nBytes);
    CHECK(cudaMemset(dC.array, 0.0, nBytes));
    start = clock();
    matrixMulGlobal<<<grid, block>>>(dA, dB, dC);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("matrix multiplication on GPU\t: %f ms\n", time * 1000.0);
    CHECK(cudaMemcpy(hC_gpu.array, dC.array, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hC, hC_gpu);

    // compute on GPU; shared memory
    memset(hC_gpu.array, 0.0, nBytes);
    CHECK(cudaMemset(dC.array, 0.0, nBytes));
    start = clock();
    matrixMulShared<<<grid, block>>>(dA, dB, dC);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("matrix multiplication on GPU + shared memory: %f ms\n", time * 1000.0);
    CHECK(cudaMemcpy(hC_gpu.array, dC.array, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hC, hC_gpu);

    CHECK(cudaFree(dA.array));
    CHECK(cudaFree(dB.array));
    CHECK(cudaFree(dC.array));
    free(hA.array);
    free(hB.array);
    free(hC.array);
    free(hC_gpu.array);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

/**********CUDA kernels**********/

__global__ void matrixMulGlobal(Matrix A, Matrix B, Matrix C) {
    int rowA = blockIdx.y * blockDim.y + threadIdx.y;
    int colB = blockIdx.x * blockDim.x + threadIdx.x;
    int value = 0;
    for (int i = 0; i < A.width; i++) 
        value += A.array[rowA * A.width + i] * B.array[i * B.width + colB];
    C.array[rowA * C.width + colB] = value;
}

__device__ Matrix getSubMatrix(Matrix m, int x, int y) {
    Matrix sub;
    sub.width = BLOCKSIZE;
    sub.height = BLOCKSIZE;
    sub.stride = m.stride;
    sub.array = &m.array[m.stride * BLOCKSIZE * y + BLOCKSIZE * x];
    return sub;
}

__global__ void matrixMulShared(Matrix A, Matrix B, Matrix C) {
    int value = 0;
    Matrix Csub = getSubMatrix(C, blockIdx.x, blockIdx.y);

    // split sub-matrix to squares of size BLOCKSIZE*BLOCKSIZE
    // Q: why not process the whole sub-matrix (e.g. As of size BLOCKSIZE*A.width)?
    // A: shared memory is limited
    for (int i = 0; i < A.width / BLOCKSIZE; i++) {
        Matrix Asub = getSubMatrix(A, i, blockIdx.y);
        Matrix Bsub = getSubMatrix(B, blockIdx.x, i);

        __shared__ int As[BLOCKSIZE][BLOCKSIZE];
        __shared__ int Bs[BLOCKSIZE][BLOCKSIZE];
        As[threadIdx.y][threadIdx.x] = Asub.array[threadIdx.y * Asub.stride + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = Bsub.array[threadIdx.y * Bsub.stride + threadIdx.x];
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k++) value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    Csub.array[threadIdx.y * Csub.stride + threadIdx.x] = value;
}

/**********host functions**********/

void matrixMulHost(Matrix A, Matrix B, Matrix C) {
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < B.width; j++) {
            int value = 0;
            for (int k = 0; k < A.width; k++)
                value += A.array[i * A.width + k] * B.array[k * B.width + j];
            C.array[i * C.width + j] = value;
        }
    }
}

Matrix initialMatrixHost(int width, int height) {
    Matrix m;
    size_t nBytes = width * height * sizeof(int);
    m.width = width;
    m.height = height;
    m.stride = width;
    m.array = (int *)malloc(nBytes);
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) { 
            m.array[h * width + w] = (int)(rand() & 0xFF) / 100.0f;
        }
    }
    return m;
}

Matrix initialMatrixDevice(Matrix mHost, bool copy) {
    Matrix m;
    m.width = mHost.width;
    m.height = mHost.height;
    m.stride = mHost.stride;
    size_t nBytes = m.width * m.height * sizeof(int);
    CHECK(cudaMalloc((int**)&m.array, nBytes));
    if (copy) CHECK(cudaMemcpy(m.array, mHost.array, nBytes, cudaMemcpyHostToDevice));
    return m;
}

void checkResult(Matrix hostRef, Matrix gpuRef) {
    double epsilon = 1.0E-8;
    int width = hostRef.width, height = hostRef.height;
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) { 
            int idx = h * width + w;
            if (abs(hostRef.array[idx] - gpuRef.array[idx]) > epsilon) {
                printf("different on (height %d, width %d): host %d gpu %d\n", 
                    h, w, hostRef.array[idx], gpuRef.array[idx]);
                return;
            }
        }
    }
}