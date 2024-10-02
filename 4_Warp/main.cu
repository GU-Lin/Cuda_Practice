#include <iostream>
#include <cuda_runtime.h>
#include <math.h>


__global__ void memory_latency_test(int* d_data, int iterations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int value = idx;
    for (int i = 0; i < iterations; ++i) {
        value = d_data[value]; // 進行訪問
    }
    // 防止編譯器優化訪問
    if (value == -1) printf("Error\n");
}

void test1()
{
    int *d_data;
    int N = 1024 * 1024; // 測試數據大小
    int iterations = 1000000; // 訪問次數

    // 分配內存
    cudaMalloc(&d_data, N * sizeof(int));

    // 設置內存中的跳躍訪問鏈
    int *h_data = new int[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = (i + 1) % N;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // 設置 CUDA 事件來測量 kernel 的執行時間
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 記錄開始事件
    cudaEventRecord(start);

    // 啟動 kernel 測試內存延遲
    dim3 grid(1);
    dim3 block(32,32,32);
    memory_latency_test<<<grid,block >>>(d_data, iterations);

    // 記錄結束事件
    cudaEventRecord(stop);

    // 等待 kernel 完成
    cudaEventSynchronize(stop);

    // 計算延遲
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 輸出結果
    // std::cout << "Total latency time: " << milliseconds << " ms" << std::endl;
    float avg_latency_per_access_us = (milliseconds / 1000) / iterations;
    std::cout << "Average latency per access: " << avg_latency_per_access_us << " sec" << std::endl;

    // 釋放內存
    cudaFree(d_data);
    delete[] h_data;

    // 釋放事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void test2()
{
    int device;
    cudaDeviceProp prop;

    // 獲取當前使用的設備 ID
    cudaGetDevice(&device);

    // 獲取設備屬性
    cudaGetDeviceProperties(&prop, device);

    // 輸出 warp 大小
    std::cout << "Device name            : " << prop.name << std::endl;
    std::cout << "Warp size              : " << prop.warpSize << std::endl;
    std::cout << "Number of SMs          : " << prop.multiProcessorCount << std::endl;
    std::cout << "Maximum threads per SM : " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Clock rate (GHz)       : " << prop.clockRate* 1e-3f << std::endl;
    std::cout << "Global memory (MBytes) : " << (float)prop.totalGlobalMem/pow(1024,3) << std::endl;
}


int main() {
    test1();
    return 0;
}