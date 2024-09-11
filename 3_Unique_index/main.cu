#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define SIZE 4194304  // 定義要生成的隨機數數量
// #define SIZE 256  // 定義要生成的隨機數數量


__global__ void print_with_pure_threadIDx(int*src)
{
    int tid = threadIdx.x;
    printf("blockIDx %d and threadIDx %d, data %d \n", blockIdx.x , threadIdx.x,  src[tid]);
}

// Block num  : 4
// Thread num : 4
__global__ void print_unique_idx_1D(int *src)
{
    int tid = threadIdx.x;
    int gid = blockDim.x*blockIdx.x + tid;
    printf("blockIDx %d and threadIDx %d, data %d \n", blockIdx.x , threadIdx.x,  src[gid]);
    
}

// Block num  : 2, 2
// Thread num : 4
__global__ void print_unique_idx_2D(int *src)
{
    int tid = threadIdx.x;
    int offset_b = blockDim.x*blockIdx.x;
    int offset_r = blockDim.x*gridDim.x*blockIdx.y;
    int gid = offset_r + offset_b + tid;
    printf("BlockIDx.x %d , blockIDx.y %d, threadIDx %d, data %d  \n", blockIdx.x, blockIdx.y , threadIdx.x, src[gid]);
}

// Block num  : 2, 2
// Thread num : 2, 2
__global__ void print_unique_idx_2D_2D(int *src)
{
    int tid = blockDim.x*threadIdx.y + threadIdx.x;
    int thread_per_block =  blockDim.x*blockDim.y;
    int offset_b = thread_per_block*blockIdx.x;

    int thread_per_row = thread_per_block*gridDim.x;
    int offset_r = thread_per_row*blockIdx.y;

    int gid = offset_r + offset_b + tid;
    printf("BlockIDx.x %d , blockIDx.y %d, threadIDx.x %d threadIDX.y %d data %d\n", blockIdx.x, blockIdx.y , threadIdx.x, threadIdx. y,src[gid]);

}


__global__ void sum_for_three_array_gpu(int* src1, int* src2, int* src3, int* src4)
{
    int tid = threadIdx.x;
    int gid = blockDim.x*blockIdx.x + tid;
    src4[gid] = src1[gid] + src2[gid] + src3[gid];
}

void sum_for_three_array_cpu(int* src1, int* src2, int* src3, int* src4)
{
    for(int i = 0; i < SIZE; i++)
    {
        src4[i] = src1[i] + src2[i] + src3[i];
    }
}

bool sum_check_for_Device(int* host_v, int* device_v)
{
    for(int i = 0; i < SIZE; i++){
        if(host_v[i] != device_v[i]){
            return false;
        }
    }
    return true;
}

void test1()
{
    int size = 8;
    int byte_size = sizeof(int) * size;
    int host_data[] = {8,10,2,5,4,23,9,7};
    int *device_data;
    cudaMalloc((void**)&device_data, byte_size);
    cudaMemcpy(device_data, host_data, byte_size, cudaMemcpyHostToDevice);
    dim3 block_per_grid(2,2);
    dim3 thread_per_block(2);
    // print_with_pure_threadIDx<<<block_per_grid, thread_per_block>>>(device_data);
    print_unique_idx_2D<<<block_per_grid, thread_per_block>>>(device_data);
    cudaDeviceSynchronize();
    cudaFree(device_data);

    cudaDeviceReset();

}

void test2()
{
    int size = 16;
    int byte_size = sizeof(int) * size;
    int host_data[] = {50, 26, 8, 9, 31, 64, 51, 20, -2, -10, 23, 7, 5, 10, 0, 1};
    int *device_data;
    cudaMalloc((void**)&device_data, byte_size);
    cudaMemcpy(device_data, host_data, byte_size, cudaMemcpyHostToDevice);    
    dim3 grid_2d_2d(2,2);
    dim3 block_2d_2d(2,2);
    print_unique_idx_2D_2D<<<grid_2d_2d, block_2d_2d>>>(device_data);

    // free(host_data);
    cudaFree(device_data);
    cudaDeviceSynchronize();
    cudaDeviceReset();
}

void test3()
{
    srand( time(NULL) );
    clock_t start_host, end_host;
    clock_t start_device, end_device;
    cudaError error;
    int size_vector = sizeof(int) * SIZE;

    // Malloc for CPU
    int *random_numbers1 = (int*)malloc(size_vector);
    int *random_numbers2 = (int*)malloc(size_vector);
    int *random_numbers3 = (int*)malloc(size_vector);
    int *random_numbers4 = (int*)malloc(size_vector);
    int *result_from_device = (int*)malloc(size_vector);

    int *random_numbersGPU1;
    int *random_numbersGPU2;
    int *random_numbersGPU3;
    int *random_numbersGPU4;
    
    // 生成隨機數並存入陣列
    for (int i = 0; i < SIZE; i++) {
        random_numbers1[i] = rand();  // 生成隨機數
        random_numbers2[i] = rand();  // 生成隨機數
        random_numbers3[i] = rand();  // 生成隨機數
        random_numbers4[i] = 0;
    }
    
    // Malloc for GPU

    error = cudaMalloc((int**)&random_numbersGPU1, size_vector);
    error = cudaMalloc((int**)&random_numbersGPU2, size_vector);
    error = cudaMalloc((int**)&random_numbersGPU3, size_vector);
    error = cudaMalloc((int**)&random_numbersGPU4, size_vector);
    if(error != cudaError::cudaSuccess){
        printf("Cuda Malloc Error");
        fprintf(stderr,"Error : %s \n", cudaGetErrorString(error));
        return ;
    }

    // Copy data from cpu -> gpu
    cudaMemcpy(random_numbersGPU1, random_numbers1, size_vector, cudaMemcpyHostToDevice);
    cudaMemcpy(random_numbersGPU2, random_numbers2, size_vector, cudaMemcpyHostToDevice);
    cudaMemcpy(random_numbersGPU3, random_numbers3, size_vector, cudaMemcpyHostToDevice);
    cudaMemcpy(random_numbersGPU4, random_numbers4, size_vector, cudaMemcpyHostToDevice);

    // Compute by cpu
    start_host = clock();
    sum_for_three_array_cpu(random_numbers1,random_numbers2,random_numbers3,random_numbers4);
    end_host = clock();
    printf("Cpu done\n");
    int nx = 512;
    dim3 grid(nx);
    dim3 block(SIZE/nx);
    // Compute by cpu
    start_device = clock();
    sum_for_three_array_gpu<<<grid, block>>>(random_numbersGPU1,random_numbersGPU2,random_numbersGPU3,random_numbersGPU4);
    end_device = clock();

    cudaDeviceSynchronize();
    printf("Gpu done\n");

    // Copy data from gpu -> cpu
    cudaMemcpy(result_from_device, random_numbersGPU4, size_vector, cudaMemcpyDeviceToHost);
    printf("Start to check\n");
    bool check = sum_check_for_Device(random_numbers4, result_from_device);
    
    // Check true and false
    printf("Valid check %s\n", check? "true":"false");
    
    // Print time
    printf("Time of CPU : %4.6f\n", (double)(double)(end_host-start_host)/CLOCKS_PER_SEC);
    printf("Time of GPU : %4.6f\n", (double)(double)(end_device-start_device)/CLOCKS_PER_SEC);

    // Free
    cudaFree(random_numbersGPU1);
    cudaFree(random_numbersGPU2);
    cudaFree(random_numbersGPU3);
    cudaFree(random_numbersGPU4);
    free(random_numbers1);
    free(random_numbers2);
    free(random_numbers3);
    free(random_numbers4);
    free(result_from_device);
    cudaDeviceReset();
}

int main(){

    test2();
    return 0;

}

