#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>


__global__ void hello_cuda(){
    printf("Hello CUDA world \n");
}

__global__ void print_threadIdx(){
    printf("gridIDx.x : %d, gridIDx.y : %d\n",
    blockIdx.x, blockIdx.y);
}

int main(){
    dim3 block(4,4,4);
    int nx = 16, ny = 16;
    dim3 grid(1,2);
    print_threadIdx<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}