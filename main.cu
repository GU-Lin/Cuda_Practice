#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>


__global__ void hello_cuda(){
    printf("Hello CUDA world \n");
}

__global__ void print_threadIdx(){
    printf("threadIDx.x : %d, threadIDx.y : %d, threadIDx.z : %d\n",
    threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(){
    dim3 block(8,8);
    int nx = 16, ny = 16;
    dim3 grid(nx/block.x,ny/block.y);
    print_threadIdx<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}