#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void) {
    printf("blockIdx (%d %d %d) threadIdx: (%d %d %d) gridDim (%d %d %d) blockDim (%d %d %d)\n",
            blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
            gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
}

int main(int argc, char **argv) {
    int nElem = 6;

    dim3 block(3); // 1-D block containing 3 threads
    dim3 grid((nElem+block.x-1)/block.x); // grid size is rounded up to the multiple of block size

    // check grid and block dim on the host side
    // unused fields will be initialized to 1
    printf("Host: check grid and block dim\n");
    printf("grid: x %d y %d z %d\n", grid.x, grid.y, grid.z);
    printf("block: x %d y %d z %d\n\n", block.x, block.y, block.z);

    // check grid and block dim from the device size
    printf("Device: check grid and block dim\n");
    checkIndex<<<grid, block>>>();

    cudaDeviceReset();

    return 0;
}
