#include <stdio.h>

// __global__ tell the compiler that the function will be called from CPU and executed from GPU
__global__ void helloFromGPU(void) {
    int idx = threadIdx.x;
    printf("GPU hello world on thread # %d\n", idx);
}

int main(void) {
    helloFromGPU <<<1, 10>>>(); // 10 threads will execute the kernel
    cudaDeviceReset(); // clean up all resources associated with the current device
    return 0;
}