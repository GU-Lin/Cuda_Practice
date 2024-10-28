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
    if(flag)
    {
        status = cudaMemcpy(deviceMatrix, hostMatrix.data(), size * sizeof(T), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            cudaFree(deviceMatrix); // 釋放已分配的設備內存
            throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(status)));
        }
    }
}
template void transferVectorToCUDA<double>(const Eigen::Matrix<double, Eigen::Dynamic, 1>&, double*&, std::size_t, bool);
template void transferVectorToCUDA<float>(const Eigen::Matrix<float, Eigen::Dynamic, 1>&, float*&, std::size_t, bool);

