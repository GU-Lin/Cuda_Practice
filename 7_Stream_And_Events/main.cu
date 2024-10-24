#include <iostream>
#include <cuda_runtime.h>


__global__ void addOneKernel(int* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] += 1;
    }
}


__global__ void multiplyByTwoKernel(int* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] *= 2;
    }
}

void test_Stream()
{
    const int arraySize = 1<<18;
    const int byteSize = arraySize * sizeof(int);

    // Allocate memory for host and device
    int* d1_data, *d2_data;
    int* h1_data, *h2_data;
    cudaMalloc(&d1_data, byteSize);
    cudaMalloc(&d2_data, byteSize);
    cudaMallocHost(&h1_data, byteSize);
    cudaMallocHost(&h2_data, byteSize);
    // Initial
    for (int i = 0; i < arraySize; ++i) {
        h1_data[i] = i;
        h2_data[i] = i;
    }

    // Construct 2 cuda stream
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    // Set up grid, block
    int blockSize = 32;
    int gridSize = (arraySize + blockSize - 1) / blockSize;
    // Async from host to device
    cudaMemcpyAsync(d1_data, h1_data, byteSize, cudaMemcpyHostToDevice, stream1);
    // kernel function1 with stream1
    addOneKernel<<<gridSize, blockSize, 0, stream1>>>(d1_data, arraySize);
    cudaMemcpyAsync(h1_data, d1_data, byteSize, cudaMemcpyDeviceToHost, stream1);
    // kernel function2 with stream2
    cudaMemcpyAsync(d2_data, h2_data, byteSize, cudaMemcpyHostToDevice, stream2);
    addOneKernel<<<gridSize, blockSize, 0, stream2>>>(d2_data, arraySize);
    cudaMemcpyAsync(h2_data, d2_data, byteSize, cudaMemcpyDeviceToHost, stream2);
    // Wait
    cudaStreamSynchronize(stream1);
    cudaStreamDestroy(stream1); 
    cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream2);
    // Free
    cudaFree(d1_data);
    cudaFree(d2_data);
    cudaFreeHost(h1_data);
    cudaFreeHost(h2_data);
}

void test_Stream_With_Events()
{
    const int arraySize = 1<<18;
    const int byteSize = arraySize * sizeof(int);

    // Allocate memory for host and device
    int* d1_data, *d2_data, *d3_data;
    int* h1_data, *h2_data, *h3_data;
    cudaMalloc(&d1_data, byteSize);
    cudaMalloc(&d2_data, byteSize);
    cudaMalloc(&d3_data, byteSize);
    cudaMallocHost(&h1_data, byteSize);
    cudaMallocHost(&h2_data, byteSize);
    cudaMallocHost(&h3_data, byteSize);

    // Initial
    for (int i = 0; i < arraySize; ++i) {
        h1_data[i] = i;
        h2_data[i] = i;
        h3_data[i] = i;
    }

    // Construct Stream
    cudaStream_t stream[3];
    for(int i = 0; i < 3; i++)
    {
        cudaStreamCreate(&stream[i]);
    }
    
    // Construct Event
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

    // Set kernel config
    int blockSize = 32;
    int gridSize = (arraySize + blockSize - 1) / blockSize;
    cudaMemcpyAsync(d1_data, h1_data, byteSize, cudaMemcpyHostToDevice, stream[0]);
    addOneKernel<<<gridSize, blockSize, 0, stream[0]>>>(d1_data, arraySize);
    // Record and set wait
    cudaEventRecord(event, stream[0]);
    cudaStreamWaitEvent(stream[2],event,0);
    cudaMemcpyAsync(h1_data, d1_data, byteSize, cudaMemcpyDeviceToHost, stream[0]);
    // Second Stream
    cudaMemcpyAsync(d2_data, h2_data, byteSize, cudaMemcpyHostToDevice, stream[1]);
    addOneKernel<<<gridSize, blockSize, 0, stream[1]>>>(d2_data, arraySize);
    cudaMemcpyAsync(h2_data, d2_data, byteSize, cudaMemcpyDeviceToHost, stream[1]);
    // Third Stream
    cudaMemcpyAsync(d3_data, h3_data, byteSize, cudaMemcpyHostToDevice, stream[2]);
    addOneKernel<<<gridSize, blockSize, 0, stream[2]>>>(d3_data, arraySize);
    cudaMemcpyAsync(h3_data, d3_data, byteSize, cudaMemcpyDeviceToHost, stream[2]);

    // Synchronize and destroy stream
    for(int i = 0; i < 3; i++)
    {
        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]); 
    }

    // Destroy Event
    cudaEventDestroy(event);
    
    // Free
    cudaFree(d1_data);
    cudaFree(d2_data);
    cudaFree(d3_data);
    cudaFreeHost(h1_data);
    cudaFreeHost(h2_data);
    cudaFreeHost(h3_data);
}

int main() {


    test_Stream_With_Events();
    return 0;
}
