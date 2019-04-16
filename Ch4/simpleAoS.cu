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

typedef struct innerStruct {
    float x;
    float y;
} innerStruct;

void initialInnerStruct(innerStruct *ip, int size);
void incrementInnerStructOnHost(innerStruct *input, innerStruct *output, const int n);
void checkResult(innerStruct *hostRef, innerStruct *gpuRef, const int n);

__global__ void incrementInnerStruct(innerStruct *input, innerStruct *output, const int n);

int main(int argc, char **argv) {
    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);
    clock_t start, end;

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // allocate host memory
    innerStruct *h_in      = (innerStruct *)malloc(nBytes);
    innerStruct *h_out     = (innerStruct *)malloc(nBytes);
    innerStruct *h_out_gpu = (innerStruct *)malloc(nBytes);
    initialInnerStruct(h_in, nElem);

    // compute on CPU
    incrementInnerStructOnHost(h_in, h_out, nElem);

    // allocate device memory
    innerStruct *d_in, *d_out;
    CHECK(cudaMalloc((innerStruct**)&d_in, nBytes));
    CHECK(cudaMalloc((innerStruct**)&d_out, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

    // execute kernel
    int blocksize = 128;
    if (argc > 1) blocksize = atoi(argv[1]);
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);
    start = clock();
    incrementInnerStruct<<<grid, block>>>(d_in, d_out, nElem);
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

__global__ void incrementInnerStruct(innerStruct *input, innerStruct *output, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        innerStruct tmp = input[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        output[i] = tmp;
    }
}

/**********host functions**********/

void initialInnerStruct(innerStruct *ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i].x = (float)(rand() & 0xFF) / 100.0f;
        ip[i].y = (float)(rand() & 0xFF) / 100.0f;
    }
    return;
}

void incrementInnerStructOnHost(innerStruct *input, innerStruct *output, const int n) {
    for (int i = 0; i < n; i++) {
        output[i].x = input[i].x + 10.f;
        output[i].y = input[i].y + 20.f;
    }
    return;
}

void checkResult(innerStruct *hostRef, innerStruct *gpuRef, const int n) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < n; i++) {
        if (abs(hostRef[i].x - gpuRef[i].x) > epsilon) {
            match = 0;
            printf("different on x field of the %dth element: host %f gpu %f\n", 
                i, hostRef[i].x, gpuRef[i].x);
            break;
        }

        if (abs(hostRef[i].y - gpuRef[i].y) > epsilon) {
            match = 0;
            printf("different on y field of the %dth element: host %f gpu %f\n", 
                i, hostRef[i].y, gpuRef[i].y);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}
