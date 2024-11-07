#include "myOperation.hpp"
#include <chrono>
int num = 128;

__global__ void matrixKernel(double* d_matrix, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows * cols) {
        // 這裡可以對矩陣進行操作，例如 d_matrix[idx] = d_matrix[idx] * 2.0;
        printf("Value %d is %f\n", idx ,d_matrix[idx]);
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

template<typename MatrixType, typename VectorType>
void CGMethodCPU(const MatrixType &h_A, const VectorType &h_b, VectorType &h_xout)
{
    // Construct 10x1 vector x0 by Eigen
    Eigen::VectorXd h_x;// = h_b;
    h_x = h_b;

    // // Construct 10x1 vector xk, rk, bk
    Eigen::VectorXd h_xk, h_rk, h_pk;
    h_xk = h_x;
    h_rk = h_b - h_A*h_xk;
    h_pk = h_rk;
    double up = 0.0;
    int i = 0;
    Eigen::VectorXd Ap ;
    auto start = std::chrono::high_resolution_clock::now();
    while(true)
    {
        double low = h_rk.dot(h_rk);
        Ap = h_A*h_pk;
        double ak = low/(h_pk.dot(Ap));
        h_xk = h_xk + ak*h_pk;
        h_rk = h_rk - ak*Ap;
        up = h_rk.dot(h_rk);
        // std::cout << sqrt(up) << std::endl;
        if(sqrt(up) < 1e-6) break;
        double bk = up/low;
        h_pk = h_rk + bk*h_pk;
        i++;   
        // break;
    }
    // h_xout = h_x;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Cpu time: " << elapsed.count() << " ms\n";
    std::cout << "Res is " << sqrt(up) << std::endl;
    std::cout << "Num is " << i << std::endl;
}

void sumCPU(double* v, int v_num, double *res)
{
    for(int i = 0; i < v_num; i++)
    {
        *res += v[i];
    }
}

template<typename MatrixType, typename VectorType>
void CGMethodGPU(const MatrixType &h_A, const VectorType &h_b, VectorType &h_xout)
{
    int rows = num;
    int cols = num;
    Eigen::VectorXd h_one(num);
    // Cuda Malloc
    double *d_A, *d_b, *d_x, *d_p;
    double *d_r, *d_Ap, *d_temp, *d_reduce;
    double *d_low, *d_up, *d_ak, *d_bk, *d_value;
    d_A = nullptr;
    d_b = nullptr;  // 初始化設備指標
    d_x = nullptr;

    // Set up grid, block
    int block_num = 64;
    int grid_num = (num+block_num-1)/block_num;
    int grid_para_num = (num+1)/(block_num*2);
    std::cout << "Grid_para_num is " << grid_para_num << std::endl;
    dim3 block(block_num);
    dim3 grid(grid_num);
    dim3 grid_para(grid_para_num);
    transferMatrixToCUDA(h_A, d_A, h_A.rows(), h_A.cols());
    transferVectorToCUDA(h_b, d_b, h_b.rows(),true); // 傳送 b，因為它是列向量，所以 cols = 1 
    transferVectorToCUDA(h_one,d_temp,h_one.rows(),true);
    cudaMalloc(&d_x, rows * sizeof(double));
    cudaMalloc(&d_Ap, rows * sizeof(double));
    cudaMalloc(&d_temp, rows * sizeof(double));
    cudaMalloc(&d_reduce, grid_para_num * sizeof(double));
    cudaMalloc(&d_p, rows * sizeof(double));
    cudaMalloc(&d_r, rows * sizeof(double));
    cudaMalloc(&d_low, 1 * sizeof(double));
    cudaMalloc(&d_up, 1 * sizeof(double));
    cudaMalloc(&d_ak, 1 * sizeof(double));
    cudaMalloc(&d_bk, 1 * sizeof(double));
    cudaMalloc(&d_value, 1*sizeof(double));

    double *h_reduce = new double[grid_para.x];
    double *h_res_temp = new double[rows];
    double h_value = 1.0;
    cudaMemcpy(d_value, &h_value, sizeof(double), cudaMemcpyHostToDevice);
    int step = 0;
    //Initial h_rx, h_xk, h_pk
    // Initial d_x
    transferVectorToCUDA(h_xout, d_x, h_xout.rows(),true);
    // Initial d_r
    matrixMultiVector<<<grid, block>>>(d_A, d_x, d_temp, rows, cols);
    vectorMinus<<<grid,block>>>(d_b, d_temp, d_value, d_r, rows);
    // Initial d_p
    cudaMemcpy(d_p, d_r, rows*sizeof(double), cudaMemcpyDeviceToDevice);

    // Start Conjugate Gradient
    auto start = std::chrono::high_resolution_clock::now();
    while(true)
    {
        // double low = h_rk.dot(h_rk);
        double h_low = 0.0;
        double h_value = 0.0;
        double h_up = 0.0;
        vectorDotElementWise<<<grid, block>>>(d_r,d_r,d_temp, rows);
        vectorDotValuePara<<<grid_para, block>>>(d_temp, d_reduce, rows);
        cudaMemcpy(h_reduce, d_reduce, grid_para.x*sizeof(double),cudaMemcpyDeviceToHost);
        sumCPU(h_reduce,grid_para.x, &h_low);

        // Ap = h_A*h_pk;
        matrixMultiVector<<<grid, block>>>(d_A, d_p, d_Ap, rows, cols);

        // double ak = low/(h_pk.dot(Ap));
        vectorDotElementWise<<<grid, block>>>(d_p, d_Ap, d_temp, rows);
        vectorDotValuePara<<<grid_para, block>>>(d_temp, d_reduce, rows);
        cudaMemcpy(h_reduce, d_reduce, grid_para.x*sizeof(double),cudaMemcpyDeviceToHost);
        sumCPU(h_reduce,grid_para.x, &h_value);

        // h_ak = h_low/h_value;
        double h_ak = h_low/h_value;
        cudaMemcpy(d_ak, &h_ak, sizeof(double), cudaMemcpyHostToDevice);

        // h_xk = h_xk + ak*h_pk;
        vectorAdd<<<grid, block>>>(d_x, d_p, d_ak, d_x, rows);

        // h_rk = h_rk - ak*Ap;
        vectorMinus<<<grid, block>>>(d_r, d_Ap, d_ak, d_r, rows);

        // up = h_rk.dot(h_rk);
        vectorDotElementWise<<<grid, block>>>(d_r, d_r, d_temp, rows);
        vectorDotValuePara<<<grid_para, block>>>(d_temp, d_reduce, rows);
        cudaMemcpy(h_reduce, d_reduce, grid_para.x*sizeof(double),cudaMemcpyDeviceToHost);
        sumCPU(h_reduce,grid_para.x, &h_up);

        if(sqrt(h_up) < 1e-6)
        {
            std::cout << "Res is " << sqrt(h_up) << std::endl;
            break;
        }
        // double bk = up/low;
        double h_bk = h_up/h_low;
        cudaMemcpy(d_bk, &h_bk, sizeof(double), cudaMemcpyHostToDevice);

        // h_pk = h_rk + bk*h_pk;
        vectorAdd<<<grid, block>>>(d_r,d_p,d_bk,d_p,rows);
        step++;   
        // break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Gpu time: " << elapsed.count() << " ms\n";
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_p);
    cudaFree(d_r);
    cudaFree(d_Ap);
    cudaFree(d_temp);
    cudaFree(d_reduce);
    cudaFree(d_up);
    cudaFree(d_low);
    cudaFree(d_ak);
    cudaFree(d_bk);
    cudaFree(d_value);
}


void test_CG()
{
    int rows = num;
    int cols = num;

    // Construct 100x100 matrix A by Eigen
    Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(rows, cols).cwiseAbs();
    h_A = h_A * h_A.transpose();

    // Construct 100x1 vector b by Eigen
    Eigen::VectorXd h_b = Eigen::VectorXd::Random(rows);
    Eigen::VectorXd h_x = h_b;
    // Eigen::VectorXd h_one(num);
    // 在這裡調用您的 CGMethodCPU 函數（確保其正確實現）
    CGMethodCPU(h_A, h_b, h_x);
    CGMethodGPU(h_A, h_b, h_x);
}


// 主程式
int main() {
    
    test_CG();
    return 0;
}
