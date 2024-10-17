#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math.h>
#include <random>
#include <chrono>

#define SHARED_SIZE 32
#define WIDTH 4096
#define HEIGHT 4096


int cpu_sum(int *vector_h,int size)
{
    int sum = 0;
    for(int i = 0; i < size; i++)
    {
        sum += vector_h[i];
    }
    return sum;
}


__global__ void test_shared_static(int* input, int* output, int size) 
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int shared_mem[SHARED_SIZE];
    if(gid < size)
    {
        shared_mem[tid] = input[gid];
        output[gid] = shared_mem[tid]+1;
    }
}

__global__ void test_shared_dynamic(int* input, int* output, int size) 
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ int shared_mem[];
    if(gid < size)
    {
        shared_mem[tid] = input[gid];
        output[gid] = shared_mem[tid]+1;
    }
}

void shared_sample()
{
    int size = 1 << 22;
    int byte_size = sizeof(int)*size;
    int *h_ref = (int*)malloc(byte_size);

    std::random_device rd;  // 硬體隨機數生成器，作為種子
    std::mt19937 gen(rd()); // Mersenne Twister 隨機數引擎
    int min = 0;
    int max = 20;
    std::uniform_int_distribution<> distrib(min, max);
    for (int i = 0; i < size; i++)
    {
        h_ref[i] = distrib(gen);  // 生成隨機數
        // printf("%d\n",vector_h[i]);
    }

    int *d_results;
    int *d_input;
    int *d_output;
    cudaMalloc((void**)&d_results, byte_size);
    cudaMalloc((void**)&d_input, byte_size);
    cudaMalloc((void**)&d_output, byte_size);
    cudaMemset(d_results, 0, byte_size);
    cudaMemset(d_output, 0, byte_size);
    cudaMemcpy(d_input,h_ref, byte_size, cudaMemcpyHostToDevice);
   

    dim3 block(SHARED_SIZE,SHARED_SIZE);
    dim3 grid(WIDTH/SHARED_SIZE, HEIGHT/SHARED_SIZE);

    test_shared_static<<<grid, block>>>(d_input, d_output, size);
    cudaMemset(d_output, 0, byte_size);
    test_shared_dynamic<<<grid, block, sizeof(int)* SHARED_SIZE>>>(d_input, d_output, size);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_results);
    free(h_ref);
}


__global__ void transpose_Naive(int* input, int* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check boundary
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

__global__ void transpose_SHEM(int* input, int* output, int width, int height) {
    __shared__ int shared_mem[SHARED_SIZE][SHARED_SIZE];

    int x = blockIdx.x * SHARED_SIZE + threadIdx.x;
    int y = blockIdx.y * SHARED_SIZE + threadIdx.y;

    // Copy data from global memory to shared memory
    if (x < width && y < height) {
        shared_mem[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    // global index
    int transposed_x = blockIdx.y * SHARED_SIZE + threadIdx.x;
    int transposed_y = blockIdx.x * SHARED_SIZE + threadIdx.y;

    // Copy data to global memory
    if (transposed_x < height && transposed_y < width) {
        output[transposed_y * height + transposed_x] = shared_mem[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_SHEM_padding(int* input, int* output, int width, int height) {
    __shared__ int shared_mem[SHARED_SIZE][SHARED_SIZE + 1]; // Avoid bank conflict

    int x = blockIdx.x * SHARED_SIZE + threadIdx.x;
    int y = blockIdx.y * SHARED_SIZE + threadIdx.y;

    // Copy data from global memory to shared memory
    if (x < width && y < height) {
        shared_mem[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    // global index
    int transposed_x = blockIdx.y * SHARED_SIZE + threadIdx.x;
    int transposed_y = blockIdx.x * SHARED_SIZE + threadIdx.y;

    // Copy data to global memory
    if (transposed_x < height && transposed_y < width) {
        output[transposed_y * height + transposed_x] = shared_mem[threadIdx.x][threadIdx.y];
    }
}


double test()
{
    int* h_in_matrix = new int[WIDTH * HEIGHT];
    int* h_out_matrix = new int[WIDTH * HEIGHT];
    int *d_input_matrix, *d_output_matrix;

    // Initial
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            h_in_matrix[i * WIDTH + j] = rand() % 10;
        }
    }

    // cudaMalloc and check
    cudaError_t err = cudaMalloc(&d_input_matrix, WIDTH * HEIGHT * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for input matrix: " 
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaMalloc(&d_output_matrix, WIDTH * HEIGHT * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for output matrix: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_matrix);
        return -1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_input_matrix, h_in_matrix, WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying input matrix to device: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_matrix);
        cudaFree(d_output_matrix);
        return -1;
    }

    // Allocate block and grid
    dim3 block(SHARED_SIZE, SHARED_SIZE);
    dim3 grid((WIDTH + SHARED_SIZE - 1) / SHARED_SIZE, (HEIGHT + SHARED_SIZE - 1) / SHARED_SIZE);

    // Launch kernel
    auto start = std::chrono::high_resolution_clock::now();
    transpose_SHEM_padding<<<grid, block>>>(d_input_matrix, d_output_matrix, WIDTH, HEIGHT);
    err = cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    if (err != cudaSuccess) {
        std::cerr << "Error synchronizing device: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_matrix);
        cudaFree(d_output_matrix);
        return -1;
    }

    // Copy data from device to host
    err = cudaMemcpy(h_out_matrix, d_output_matrix, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error copying output matrix to host: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_matrix);
        cudaFree(d_output_matrix);
        return -1;
    }

    // Validation
    bool correct = true;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            if (h_out_matrix[j * HEIGHT + i] != h_in_matrix[i * WIDTH + j]) {
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }

    // if (correct) {
    //     std::cout << "矩陣轉置正確!" << std::endl;
    // } else {
    //     std::cout << "矩陣轉置錯誤!" << std::endl;
    // }

    // 釋放設備記憶體
    cudaFree(d_input_matrix);
    cudaFree(d_output_matrix);

    // 釋放主機記憶體
    delete[] h_in_matrix;
    delete[] h_out_matrix;
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

int main()
{
    int num = 10;
    double average_time = 0.0;
    for(int i = 0; i < num; i++)
    {
        double temp = test();
        if(i==0)continue;
        average_time += (temp/(num-1));
        std::cout << "Time: " << (temp/(num-1))/1000 << " s\n";
    }
    std::cout << "-------------------" << std::endl;
    int data_size = WIDTH * HEIGHT * sizeof(int);
    float bandwidth = 2*data_size / (average_time/1000*1024*1024*1024);
    std::cout << "Time: " << average_time/1000 << " s\n";
    std::cout << "Memory Bandwidth: " << bandwidth << " GB/s\n";
    
    return 0;
}
