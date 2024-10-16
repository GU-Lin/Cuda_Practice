#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <random>
#include <chrono>
#define SIZE  4194304

int cpu_sum(int *vector_h)
{
    int sum = 0;
    for(int i = 0; i < SIZE; i++)
    {
        sum += vector_h[i];
    }
    return sum;
}


__global__ void reduce0(int *g_idata, int *g_odata) 
{
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) 
    {
        if (tid % (2*s) == 0) {
        sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce1(int *g_idata, int *g_odata) 
{
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) 
    {
        int index = 2 * s * tid;
        if (index < blockDim.x) 
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce2(int *g_idata, int *g_odata) 
{
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce3(int *g_idata, int *g_odata) 
{
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__device__ void warpReduce(volatile int* sdata, int tid) 
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce4(int *g_idata, int *g_odata) 
{
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid < 32)
    {
        warpReduce(sdata, tid);
    } 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduce5(int *g_idata, int *g_odata) 
{
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    // if(blockDim.x>=512 && tid<256)
    // {
    //     sdata[tid] += sdata[tid+256];
    // }
    // __syncthreads();
    // if(blockDim.x>=256 && tid<128)
    // {
    //     sdata[tid] += sdata[tid+128];
    // }
    // __syncthreads();
    if(blockDim.x>=128 && tid<64)
    {
        sdata[tid] += sdata[tid+64];
    }
    __syncthreads();
    // write result for this block to global mem
    if (tid < 32)
    {
        warpReduce(sdata, tid);
    } 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void test()
{

    int size_vector = sizeof(int) * SIZE;
    int *vector_h  = (int*)malloc(size_vector);
    
    int *vector_d , *vector_temp;
    cudaMalloc(&vector_d, SIZE * sizeof(int));
    cudaMalloc(&vector_temp, SIZE * sizeof(int));
    cudaError error;
    std::random_device rd;  // 硬體隨機數生成器，作為種子
    std::mt19937 gen(rd()); // Mersenne Twister 隨機數引擎
    int min = 0;
    int max = 20;
    std::uniform_int_distribution<> distrib(min, max);
    for (int i = 0; i < SIZE; i++)
    {
        vector_h[i] = distrib(gen);  // 生成隨機數
        // printf("%d\n",vector_h[i]);
    }
    // Set up grid, block for kernel
    int block_size = 128;
    dim3 block(block_size);
    dim3 grid((SIZE/block_size));
    int temp_array_byte_size = sizeof(int)* grid.x;
    printf("Kernel launch parameters | grid.x : %d, block.x : %d \n",
    grid.x, block.x);
    // Host to device

    cudaMemcpy(vector_d,vector_h, size_vector, cudaMemcpyHostToDevice);
    // Execute 
    auto start = std::chrono::high_resolution_clock::now();
    reduce1<<<grid, block>>>(vector_d,vector_temp);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    // Device to host
    int *vector_h_ref  = (int*)malloc(temp_array_byte_size);
    cudaMemcpy(vector_h_ref,vector_temp, temp_array_byte_size, cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    int gpu_result = 0;

    for(int i = 0; i < grid.x; i++)
    {
        gpu_result += vector_h_ref[i];
    }

    int sum = 0;
    // Cpu sum
    auto start_cpu = std::chrono::high_resolution_clock::now();
    sum = cpu_sum(vector_h);
    auto end_cpu = std::chrono::high_resolution_clock::now();


    // Check cpu == gpu
    printf("CPU sum %d, GPU sum %d \n", sum, gpu_result);
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::chrono::duration<double, std::milli> elapsed_cpu = end_cpu - start_cpu;
    std::cout << "Cpu time: " << elapsed_cpu.count() << " ms\n";
    std::cout << "Gpu time: " << elapsed.count() << " ms\n";
    float data_size_gb = 2.0f * size_vector / (1024 * 1024 * 1024);
    float bandwidth = data_size_gb / (elapsed.count() / 1000.0f);
    std::cout << "Memory Bandwidth: " << bandwidth << " GB/s\n";
    // printf("Gpu tim %f ms\n",milliseconds);
    error = cudaFree(vector_d);
    error = cudaFree(vector_temp);
    free(vector_h);
}

void print_memory_information()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Memory Clock Rate (MHz): " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Bus Width (bits): " << prop.memoryBusWidth << " bits" << std::endl;
}

int main()
{
    
    for(int i = 0; i < 20; i++)
    {
        test();
    }

    
    return 0;
}
