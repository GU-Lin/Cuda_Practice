#include "myOperation.hpp"


__global__ void matrixKernel(double* d_matrix, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows * cols) {
        // 這裡可以對矩陣進行操作，例如 d_matrix[idx] = d_matrix[idx] * 2.0;
        d_matrix[idx] = d_matrix[idx] * 2.0;
    }
}

void test()
{
    const int rows = 10;
    const int cols = 10;

    // Construct 10x10 matrix by Eigen
    Eigen::Matrix<double, rows, cols> hostMatrix;
    hostMatrix.setRandom(); // 初始化為隨機值

    // Cuda Malloc
    double* d_matrix;
    size_t size = rows * cols * sizeof(double);
    cudaMalloc(&d_matrix, size);

    // Host to Device
    cudaMemcpy(d_matrix, hostMatrix.data(), size, cudaMemcpyHostToDevice);

    // Launch kernel function
    int threadsPerBlock = 128;
    int blocksPerGrid = (rows * cols + threadsPerBlock - 1) / threadsPerBlock;
    matrixKernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, rows, cols);
    cudaDeviceSynchronize(); // 等待 CUDA 核心函數完成

    // Device to Host
    Eigen::Matrix<double, rows, cols> resultMatrix;
    cudaMemcpy(resultMatrix.data(), d_matrix, size, cudaMemcpyDeviceToHost);


    std::cout << "Original Matrix:\n" << hostMatrix << std::endl;
    std::cout << "Result Matrix:\n" << resultMatrix << std::endl;

    // Free
    cudaFree(d_matrix);
}

// 主程式
int main() {
    const int num = 10;
    const int rows = num;
    const int cols = num;

    // Construct 10x10 matrix A by Eigen
    Eigen::Matrix<double, rows, cols> h_A;

    h_A.setRandom();
    h_A = h_A.cwiseAbs();
    h_A = h_A*h_A.transpose();
    std::cout << h_A << std::endl;
    // Construct 10x1 vector b by Eigen
    Eigen::Matrix<double, rows, 1> h_b;
    h_b.setRandom();

    // Construct 10x1 vector x0 by Eigen
    Eigen::Matrix<double, rows, 1> h_x;
    h_x = h_b;

    // Construct 10x1 vector xk, rk, bk
    Eigen::Matrix<double, rows, 1> h_xk, h_rk, h_pk;
    h_xk = h_x;
    h_rk = h_b - h_A*h_xk;
    h_pk = h_rk;
    double up = 0.0;
    int i = 0;
    while(true && i < num*2)
    {
        double low = h_rk.dot(h_rk);
        Eigen::Matrix<double, rows, 1>Ap = h_A*h_pk;
        double ak = low/(h_pk.dot(Ap));
        h_xk = h_xk + ak*h_pk;
        h_rk = h_rk - ak*Ap;
        up = h_rk.dot(h_rk);
        if(sqrt(up) < 1e-6) break;
        double bk = up/low;
        h_pk = h_rk + bk*h_pk;
        i++;   
    }
    std::cout << "Res is " << sqrt(up) << std::endl;
    std::cout << "Num is " << i << std::endl;
    return 0;
}
