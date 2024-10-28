#include "myOperation.hpp"
int num = 3;

__global__ void matrixKernel(double* d_matrix, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows * cols) {
        // 這裡可以對矩陣進行操作，例如 d_matrix[idx] = d_matrix[idx] * 2.0;
        printf("Value %d is %f\n", idx ,d_matrix[idx]);
    }
}

__global__ void matrixMultiVector(double* d_matrix, double* d_vector_in,double* d_vector_out, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int out_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < cols) {
        // 這裡可以對矩陣進行操作，例如 d_matrix[idx] = d_matrix[idx] * 2.0;
        double sum = 0.0;
        for(int j = 0; j < cols; j++)
        {
            sum += d_matrix[out_row*rows + j]*d_vector_in[j];
        }
        d_vector_out[out_row] = sum;
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

__global__ void vectorAdd(double* d_vector1, double* d_vector2, double* d_out, int rows) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows ) {
        // 這裡可以對矩陣進行操作，例如 d_matrix[idx] = d_matrix[idx] * 2.0;
        d_out[idx] = d_vector2[idx] + d_vector1[idx];
    }
    __syncthreads();
}

__global__ void vectorMinus(double* d_vector1, double* d_vector2, double* d_out, int rows) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows ) {
        // 這裡可以對矩陣進行操作，例如 d_matrix[idx] = d_matrix[idx] * 2.0;
        d_out[idx] = d_vector1[idx] - d_vector2[idx];
    }
    __syncthreads();
}

__global__ void vectorDotRestoreToVector(double* d_vector1, double* d_vector2, double* d_vector3, int rows) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double sum = 0.0;
    if (idx < rows ) {
        // 這裡可以對矩陣進行操作，例如 d_matrix[idx] = d_matrix[idx] * 2.0;
        d_vector3[idx] = d_vector2[idx] * d_vector1[idx];
    }
    __syncthreads();
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
    while(true && i < num*2)
    {
        double low = h_rk.dot(h_rk);
        Ap = h_A*h_pk;
        double ak = low/(h_pk.dot(Ap));
        h_xk = h_xk + ak*h_pk;
        h_rk = h_rk - ak*Ap;
        up = h_rk.dot(h_rk);
        if(sqrt(up) < 1e-6) break;
        double bk = up/low;
        h_pk = h_rk + bk*h_pk;
        i++;   
    }
    h_xout = h_x;
    std::cout << "Res is " << sqrt(up) << std::endl;
    std::cout << "Num is " << i << std::endl;
}


// 主程式
int main() {
    
        
    int rows = num;
    int cols = num;

    // Construct 100x100 matrix A by Eigen
    Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(rows, cols).cwiseAbs();
    h_A = h_A * h_A.transpose();

    // Construct 100x1 vector b by Eigen
    Eigen::VectorXd h_b = Eigen::VectorXd::Random(rows);
    Eigen::VectorXd h_x;
    double *h_res = new double(1.0);
    // 在這裡調用您的 CGMethodCPU 函數（確保其正確實現）
    // CGMethodCPU(h_A, h_b, h_x);

    // Cuda Malloc
    double *d_A, *d_b, *d_x, *d_p, *d_r, *d_Ap, *d_temp, *d_up, *d_low, *d_ak, *d_bk;
    d_A = nullptr;
    d_b = nullptr;  // 初始化設備指標
    d_x = nullptr;

    transferMatrixToCUDA(h_A, d_A, h_A.rows(), h_A.cols());
    transferVectorToCUDA(h_b, d_b, h_b.rows(),true); // 傳送 b，因為它是列向量，所以 cols = 1
    cudaMalloc(&d_x, rows * sizeof(double));
    cudaMalloc(&d_Ap, rows * sizeof(double));
    cudaMalloc(&d_temp, rows * sizeof(double));
    cudaMalloc(&d_p, rows * sizeof(double));
    cudaMalloc(&d_r, rows * sizeof(double));
    cudaMalloc(&d_low, 1 * sizeof(double));
    cudaMalloc(&d_up, 1 * sizeof(double));
    cudaMalloc(&d_ak, 1 * sizeof(double));
    cudaMalloc(&d_bk, 1 * sizeof(double));

    // Set up grid, block
    dim3 block(num);
    dim3 grid(1);
    matrixMultiVector<<<block,grid>>>(d_A,d_b,d_x,rows,cols);
    vectorDot<<<block,grid>>>(d_x,d_x,d_value,rows);
    double *h_x_res = new double[num];
    double *h_value = new double (0);
    cudaMemcpy(h_x_res, d_x, num*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_value, d_value, 1*sizeof(double), cudaMemcpyDeviceToHost);
    int i = 0;
    //Initial h_rx, h_xk, h_pk
    // Initial d_x
    transferVectorToCUDA(h_b, d_x, h_b.rows(),true);
    // Initial d_r
    matrixMultiVector<<<grid, block>>>(d_A, d_x, d_temp, rows, cols);
    vectorMinus<<<grid,block>>>(d_b, d_temp, d_r, rows);
    // Initial d_p
    cudaMemcpy(d_p, d_r, rows*sizeof(double), cudaMemcpyDeviceToDevice);
    while(true && h_res > 1e-6)
    {
    //  double low = h_rk.dot(h_rk);
        vectorDotRestoreToVector<<<block, grid>>>()
    //  Ap = h_A*h_pk;
        matrixMultiVector<<<block, grid>>>(d_A, d_x, d_Ap, rows, cols);
    //  double ak = low/(h_pk.dot(Ap));

    //  h_xk = h_xk + ak*h_pk;

    //  h_rk = h_rk - ak*Ap;

    //  up = h_rk.dot(h_rk);

    //  if(sqrt(up) < 1e-6) break;

    //  double bk = up/low;

    //  h_pk = h_rk + bk*h_pk;

    //  i++;   
    }





    // 進行其他操作（例如計算或使用 d_A 和 d_b）

    // 清理
    cudaFree(d_A);
    cudaFree(d_b);

    return 0;
}
