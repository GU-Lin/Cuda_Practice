#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void hello_cuda(){
    printf("Hello CUDA world \n");
}

__global__ void hello_cuda_with_id(){
    printf("Hello CUDA world with blockIDx.x %d, threadIDx %d \n", blockIdx.x, threadIdx.x);
}

int main(){

    dim3 grid(2);
    dim3 block(4);
    hello_cuda_with_id<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;

}