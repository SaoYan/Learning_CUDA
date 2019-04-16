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

#define LEN 1<<22

typedef struct innerArray {
    float x[LEN];
    float y[LEN];
} innerArray;

void initialInnerArray(innerArray *ip,  int size);
void incrementInnerArrayOnHost(innerArray *input, innerArray *output, const int n);
void checkResult(innerArray *hostRef, innerArray *gpuRef, const int n);

__global__ void incrementInnerArray(innerArray *input, innerArray * output, const int n);

int main(int argc, char **argv) {
    int nElem = LEN;
    size_t nBytes = sizeof(innerArray);
    clock_t start, end;

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // allocate host memory
    innerArray *h_in      = (innerArray *)malloc(nBytes);
    innerArray *h_out     = (innerArray *)malloc(nBytes);
    innerArray *h_out_gpu = (innerArray *)malloc(nBytes);
    initialInnerArray(h_in, nElem);

    // compute on CPU
    incrementInnerArrayOnHost(h_in, h_out, nElem);

    // allocate device memory
    innerArray *d_in, *d_out;
    CHECK(cudaMalloc((innerArray**)&d_in, nBytes));
    CHECK(cudaMalloc((innerArray**)&d_out, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

    // execute kernel
    int blocksize = 128;
    if (argc > 1) blocksize = atoi(argv[1]);
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);
    start = clock();
    incrementInnerArray<<<grid, block>>>(d_in, d_out, nElem);
    CHECK(cudaDeviceSynchronize()); // synchronize kernel only for debugging!
    end = clock();
    double time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("<<< %3d, %3d >>> elapsed %f ms\n", grid.x, block.x, time * 1000.0);

    // copy data back to CPU
    CHECK(cudaMemcpy(h_out_gpu, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, h_out_gpu, nElem);
    CHECK(cudaGetLastError());

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

__global__ void incrementInnerArray(innerArray *input, innerArray * output, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float tmpx = input->x[i];
        float tmpy = input->y[i];
        output->x[i] = tmpx + 10.f;
        output->y[i] = tmpy + 20.f;
    }
}

/**********host functions**********/

void initialInnerArray(innerArray *ip,  int size) {
    for (int i = 0; i < size; i++) {
        ip->x[i] = (float)( rand() & 0xFF ) / 100.0f;
        ip->y[i] = (float)( rand() & 0xFF ) / 100.0f;
    }
    return;
}

void incrementInnerArrayOnHost(innerArray *input, innerArray *output, const int n)
{
    for (int i = 0; i < n; i++) {
        output->x[i] = input->x[i] + 10.f;
        output->y[i] = input->y[i] + 20.f;
    }
    return;
}

void checkResult(innerArray *hostRef, innerArray *gpuRef, const int n) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < n; i++) {
        if (abs(hostRef->x[i] - gpuRef->x[i]) > epsilon) {
            match = 0;
            printf("different on x %dth element: host %f gpu %f\n", 
                i, hostRef->x[i], gpuRef->x[i]);
            break;
        }

        if (abs(hostRef->y[i] - gpuRef->y[i]) > epsilon) {
            match = 0;
            printf("different on y %dth element: host %f gpu %f\n", 
                i, hostRef->y[i], gpuRef->y[i]);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}