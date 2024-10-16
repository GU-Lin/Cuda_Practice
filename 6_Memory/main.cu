#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math.h>
#include <random>
#include <chrono>

#define SHARED_SIZE 16
#define M 1024
#define N 1024
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
#define IPAD 2

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
    dim3 grid(N/SHARED_SIZE, M/SHARED_SIZE);

    test_shared_static<<<grid, block>>>(d_input, d_output, size);
    cudaMemset(d_output, 0, byte_size);
    test_shared_dynamic<<<grid, block, sizeof(int)* SHARED_SIZE>>>(d_input, d_output, size);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_results);
    free(h_ref);
}

__global__ void transepose_gpu_SHEM(int* input, int* output)
{
    __shared__ int shared_mem[BLOCKSIZE_Y][BLOCKSIZE_X+1];
    const unsigned  int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// find shmem[row][col] mapped to current thread
	const unsigned int tid = blockDim.x * threadIdx.y + threadIdx.x;
	const unsigned int col = tid / blockDim.y;
	const unsigned int row = tid % blockDim.y;

	// find the global index of (sx, sy) mapped to the local shmem[row][col]
	const unsigned int sx = blockIdx.x * blockDim.x + col;
	const unsigned int sy = blockIdx.y * blockDim.y + row;

	// then copy shmem[row][col] to out matrix (sy, sx)
    if(x < N && y < M)
    {
        shared_mem[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();
    output[sx*M + sy] = shared_mem[row][col];

}

__global__ void transpose_gpu_SHEM_1(int* input, int* output, int width, int height) {
    __shared__ int shared_mem[SHARED_SIZE][SHARED_SIZE + 2]; // 避免 bank conflict，添加一個偏移量
    int x = blockIdx.x * SHARED_SIZE + threadIdx.x;
    int y = blockIdx.y * SHARED_SIZE + threadIdx.y;

    // 把資料從 global memory 複製到 shared memory
    if (x < width && y < height) {
        shared_mem[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    // 轉置座標
    int transposed_x = blockIdx.y * SHARED_SIZE + threadIdx.x;
    int transposed_y = blockIdx.x * SHARED_SIZE + threadIdx.y;

    // 把轉置後的資料寫回 global memory
    if (transposed_x < height && transposed_y < width) {
        output[transposed_y * height + transposed_x] = shared_mem[threadIdx.x][threadIdx.y];
    }
}

int test()
{
    // 主機端矩陣
    int h_in_matrix[M][N], h_out_matrix[N][M];
    int *d_input_matrix, *d_output_matrix;

    // 初始化輸入矩陣
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_in_matrix[i][j] = rand() % 10;
        }
    }

    // 分配設備記憶體，並檢查錯誤
    if (cudaMalloc(&d_input_matrix, M * N * sizeof(int)) != cudaSuccess) {
        std::cerr << "Error allocating device memory for input matrix." << std::endl;
        return -1;
    }
    if (cudaMalloc(&d_output_matrix, N * M * sizeof(int)) != cudaSuccess) {
        std::cerr << "Error allocating device memory for output matrix." << std::endl;
        cudaFree(d_input_matrix);
        return -1;
    }

    // 拷貝主機矩陣到設備記憶體，並檢查錯誤
    if (cudaMemcpy(d_input_matrix, h_in_matrix, M * N * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Error copying input matrix to device." << std::endl;
        cudaFree(d_input_matrix);
        cudaFree(d_output_matrix);
        return -1;
    }

    // 配置 grid 和 block 大小
    dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // 調用 CUDA kernel，並檢查錯誤
    transpose_gpu_SHEM_1<<<grid, block>>>(d_input_matrix, d_output_matrix, N, M);
    if (cudaPeekAtLastError() != cudaSuccess) {
        std::cerr << "Error launching the kernel." << std::endl;
        cudaFree(d_input_matrix);
        cudaFree(d_output_matrix);
        return -1;
    }

    // 等待設備完成，並檢查錯誤
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "Error synchronizing device." << std::endl;
        cudaFree(d_input_matrix);
        cudaFree(d_output_matrix);
        return -1;
    }

    // 拷貝結果回主機，並檢查錯誤
    if (cudaMemcpy(h_out_matrix, d_output_matrix, N * M * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Error copying output matrix to host." << std::endl;
        cudaFree(d_input_matrix);
        cudaFree(d_output_matrix);
        return -1;
    }

    // // 驗證結果
    // bool correct = true;
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         if (h_out_matrix[j][i] != h_in_matrix[i][j]) {
    //             correct = false;
    //             break;
    //         }
    //     }
    //     if (!correct) break;
    // }

    // if (correct) {
    //     std::cout << "矩陣轉置正確!" << std::endl;
    // } else {
    //     std::cout << "矩陣轉置錯誤!" << std::endl;
    // }

    // 釋放設備記憶體
    cudaFree(d_input_matrix);
    cudaFree(d_output_matrix);
    return 0;
}

int main()
{

    test();
    
    return 0;
}
