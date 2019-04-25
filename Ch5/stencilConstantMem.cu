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

#define RADIUS 4
#define BLOCKSIZE 256

// coeffecient
#define a0     0.00000f
#define a1     0.80000f
#define a2    -0.20000f
#define a3     0.03809f
#define a4    -0.00357f

// constant memory
__constant__ float coef[RADIUS + 1];

void initialData(float *ip, const int size);
void stencilHost(float *in, float *out, int isize);
void checkResult(float *hostRef, float *gpuRef, const int size);

__global__ void stencilGPU(float *in, float *out, const int n);

int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // set up data size
    int isize = 1 << 11;

    size_t nBytes = (isize + 2 * RADIUS) * sizeof(float);
    printf("array size: %d\n", isize);

    // allocate host memory
    float *h_in      = (float *)malloc(nBytes);
    float *h_out     = (float *)malloc(nBytes);
    float *h_out_gpu = (float *)malloc(nBytes);
    initialData(h_in, isize + 2 * RADIUS);

    // compute on CPU
    stencilHost(h_in, h_out, isize);

    // allocate device memory
    float *d_in, *d_out;
    CHECK(cudaMalloc((float**)&d_in, nBytes));
    CHECK(cudaMalloc((float**)&d_out, nBytes));
    CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

    // set up constant memory
    const float h_coef[] = {a0, a1, a2, a3, a4};
    CHECK(cudaMemcpyToSymbol(coef, h_coef, (RADIUS + 1) * sizeof(float)));

    // launch CUDA kernel
    // cudaDeviceProp info;
    // CHECK(cudaGetDeviceProperties(&info, 0));
    dim3 block(BLOCKSIZE);
    // dim3 grid(info.maxGridSize[0] < isize / block.x ? info.maxGridSize[0] : isize / block.x);
    dim3 grid(isize / block.x);
    printf("(grid, block) %d,%d \n", grid.x, block.x);
    stencilGPU<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS, isize);

    CHECK(cudaMemcpy(h_out_gpu, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, h_out_gpu, isize);

    // free memory
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    free(h_out_gpu);

    // reset device
    CHECK(cudaDeviceReset());
    return 0;
}

/**********CUDA kernels**********/

__global__ void stencilGPU(float *in, float *out, const int n) {
    // shared memory
    __shared__ float smem[BLOCKSIZE + 2 * RADIUS];

    // index to global memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < n) {
        int sidx = threadIdx.x + RADIUS; // index to shared memory for stencil calculatioin
        smem[sidx] = in[idx];            // read data from global memory into shared memory
        if (threadIdx.x < RADIUS) {      // read halo part to shared memory
            smem[sidx - RADIUS]    = in[idx - RADIUS];
            smem[sidx + BLOCKSIZE] = in[idx + BLOCKSIZE];
        }
        __syncthreads();

        float tmp = 0.0f;
        #pragma unroll
        for (int i = 1; i <= RADIUS; i++) {
            tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
        }
        out[idx] = tmp;

        idx += gridDim.x * blockDim.x;
    }
}

/**********host functions**********/

void initialData(float *ip, const int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 100.0f;
    }
}

void stencilHost(float *in, float *out, int isize)
{
    for (int i = RADIUS; i <= isize; i++) {
        out[i] = a1 * (in[i + 1] - in[i - 1])
                    + a2 * (in[i + 2] - in[i - 2])
                    + a3 * (in[i + 3] - in[i - 3])
                    + a4 * (in[i + 4] - in[i - 4]);
    }
}

void checkResult(float *hostRef, float *gpuRef, const int size) {
    double epsilon = 1.0E-6;
    for (int i = RADIUS; i < size; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }
}
