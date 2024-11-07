#include <myOperation.hpp>


template<typename T>
void transferMatrixToCUDA(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& hostMatrix, T*& deviceMatrix, std::size_t rows, std::size_t cols) 
{
    // 計算所需的字節數
    std::size_t size = rows * cols;  // 不需要乘以 sizeof(T)

    // 在設備上分配內存
    cudaError_t status = cudaMalloc((void**)&deviceMatrix, size * sizeof(T)); // 這裡正確計算內存大小
    if (status != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(status)));
    }

    // 拷貝數據到設備
    status = cudaMemcpy(deviceMatrix, hostMatrix.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(deviceMatrix); // 釋放已分配的設備內存
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(status)));
    }
}
template void transferMatrixToCUDA<double>(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>&, double*&, std::size_t, std::size_t);
template void transferMatrixToCUDA<float>(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&, float*&, std::size_t, std::size_t);

template<typename T>
void transferVectorToCUDA(const Eigen::Matrix<T, Eigen::Dynamic, 1>& hostMatrix, T*& deviceMatrix, std::size_t rows, bool flag) 
{
    // 計算所需的字節數
    std::size_t size = rows;  // 不需要乘以 sizeof(T)

    // 在設備上分配內存
    cudaError_t status = cudaMalloc((void**)&deviceMatrix, size * sizeof(T)); // 這裡正確計算內存大小
    if (status != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(status)));
    }

    // 拷貝數據到設備
    // if(flag)
    // {
    status = cudaMemcpy(deviceMatrix, hostMatrix.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(deviceMatrix); // 釋放已分配的設備內存
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(status)));
    }
    // }
}
template void transferVectorToCUDA<double>(const Eigen::Matrix<double, Eigen::Dynamic, 1>&, double*&, std::size_t, bool);
template void transferVectorToCUDA<float>(const Eigen::Matrix<float, Eigen::Dynamic, 1>&, float*&, std::size_t, bool);

__global__ void matrixMultiVector(double* d_matrix, double* d_vector_in,double* d_vector_out, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < cols) {
        
        double sum = 0.0;
        for(int j = 0; j < cols; j++)
        {
            sum += d_matrix[idx*cols + j]*d_vector_in[j];
        }
        d_vector_out[idx] = sum;
        
    }
    __syncthreads();
}

__global__ void vectorMultiScale(double* d_vector, double s,int rows) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows ) {
        // 這裡可以對矩陣進行操作，例如 d_matrix[idx] = d_matrix[idx] * 2.0;
        d_vector[idx] *= s;
    }
    __syncthreads();
}

__global__ void vectorAdd(double* d_vector1, double* d_vector2, double* scale, double* d_out, int rows) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows ) {
        // 這裡可以對矩陣進行操作，例如 d_matrix[idx] = d_matrix[idx] * 2.0;
        d_out[idx] = d_vector1[idx] + (*scale)*d_vector2[idx];
    }
    __syncthreads();
}

__global__ void vectorMinus(double* d_vector1, double* d_vector2, double* scale, double* d_out, int rows) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows ) {
        // 這裡可以對矩陣進行操作，例如 d_matrix[idx] = d_matrix[idx] * 2.0;
        d_out[idx] = d_vector1[idx] - (*scale)*d_vector2[idx];
        // printf("Num %d with %f is: %f - %f = %f\n",idx,*scale,d_vector1[idx],d_vector2[idx],d_out[idx]);
    }
    __syncthreads();
}

__global__ void vectorDotElementWise(double* d_vector1, double* d_vector2, double* d_vector3, int rows) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows ) {
        // 這裡可以對矩陣進行操作，例如 d_matrix[idx] = d_matrix[idx] * 2.0;
        d_vector3[idx] = d_vector2[idx] * d_vector1[idx];
        // printf("%d %f\n", idx, d_vector3[idx]);
    }
    __syncthreads();
}

__device__ void warpReduce(volatile double* sdata, int tid) 
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void vectorDotValue(double* d_vector_in, double* d_vector_out, int rows)
{
    extern __shared__ double sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < rows) {
        sdata[tid] = d_vector_in[i];
    } else {
        sdata[tid] = 0.0;  // 若超出范围，赋值为0
    }
    // printf("blockDim.x is %d\n",blockDim.x);
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
    if (tid == 0)
    {
        d_vector_out[blockIdx.x] = sdata[0];
        
    } 
}

__global__ void vectorDotValuePara(double* d_vector_in, double* d_vector_out, int rows)
{
    extern __shared__ double sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x*blockDim.x*2) + threadIdx.x;
    sdata[tid] = d_vector_in[i]+ d_vector_in[i+blockDim.x];
    
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
    if (tid == 0)
    {
        d_vector_out[blockIdx.x] = sdata[0];
        // printf("Output ID %d is %f\n",blockIdx.x,d_vector_out[blockIdx.x]);
    } 
}