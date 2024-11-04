#include "myOperation.hpp"
int num = 128;

__global__ void matrixKernel(double* d_matrix, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows * cols) {
        // 這裡可以對矩陣進行操作，例如 d_matrix[idx] = d_matrix[idx] * 2.0;
        printf("Value %d is %f\n", idx ,d_matrix[idx]);
    }
}

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

__global__ void scaleDivision(double* up, double* low, double* out)
{
    *out = (*up) / (*low);
}

__global__ void initializeToOne(double *d_ptr) {
    *d_ptr = 1.0;
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
        break;
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
    Eigen::VectorXd h_x = h_b;
    Eigen::VectorXd h_one(num);
    double *h_res = new double(1.0);
    // 在這裡調用您的 CGMethodCPU 函數（確保其正確實現）
    // CGMethodCPU(h_A, h_b, h_x);

    // Cuda Malloc
    double *d_A, *d_b, *d_x, *d_p;
    double *d_r, *d_Ap, *d_temp, *d_up;
    double *d_low, *d_ak, *d_bk, *d_value;
    d_A = nullptr;
    d_b = nullptr;  // 初始化設備指標
    d_x = nullptr;

    transferMatrixToCUDA(h_A, d_A, h_A.rows(), h_A.cols());
    transferVectorToCUDA(h_b, d_b, h_b.rows(),true); // 傳送 b，因為它是列向量，所以 cols = 1
    
    transferVectorToCUDA(h_one,d_temp,h_one.rows(),true);
    cudaMalloc(&d_x, rows * sizeof(double));
    cudaMalloc(&d_Ap, rows * sizeof(double));
    cudaMalloc(&d_temp, rows * sizeof(double));
    cudaMalloc(&d_p, rows * sizeof(double));
    cudaMalloc(&d_r, rows * sizeof(double));
    cudaMalloc(&d_low, 1 * sizeof(double));
    cudaMalloc(&d_up, 1 * sizeof(double));
    cudaMalloc(&d_ak, 1 * sizeof(double));
    cudaMalloc(&d_bk, 1 * sizeof(double));
    cudaMalloc(&d_value, 1*sizeof(double));
    // Set up grid, block
    int block_num = num;
    dim3 block(block_num);
    dim3 grid((num+block_num-1)/block_num);
    // dim3 grid(1);
    dim3 grid_para((num+block_num-1)/(block_num*2));
    matrixMultiVector<<<grid, block>>>(d_A,d_b,d_x,rows,cols);
    double *h_x_res = new double[rows];
    double *h_x_res1 = new double[rows];
    double h_value = 1.0;
    cudaMemcpy(d_value, &h_value, sizeof(double), cudaMemcpyHostToDevice);
    int i = 0;
    //Initial h_rx, h_xk, h_pk
    // Initial d_x
    transferVectorToCUDA(h_x, d_x, h_x.rows(),true);
    // Initial d_r
    matrixMultiVector<<<grid, block>>>(d_A, d_x, d_temp, rows, cols);
    vectorMinus<<<grid,block>>>(d_b, d_temp, d_value, d_r, rows);
    // Initial d_p
    cudaMemcpy(d_p, d_r, rows*sizeof(double), cudaMemcpyDeviceToDevice);

    // Start Conjugate Gradient
    while(1)
    {
        // double low = h_rk.dot(h_rk);
        double h_low = 0.0;
        double h_value = 0.0;
        vectorDotElementWise<<<grid, block>>>(d_r,d_r,d_temp, rows);
        vectorDotValue<<<grid, block>>>(d_temp, d_temp, rows);
        cudaMemcpy(&h_low, d_temp, sizeof(double),cudaMemcpyDeviceToHost);
        // Ap = h_A*h_pk;
        matrixMultiVector<<<grid, block>>>(d_A, d_p, d_Ap, rows, cols);
        // double ak = low/(h_pk.dot(Ap));
        vectorDotElementWise<<<grid, block>>>(d_p, d_Ap, d_temp, rows);
        vectorDotValue<<<grid, block>>>(d_temp, d_temp, rows);
        cudaMemcpy(&h_value, d_temp, sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "Value are "<<h_low << ", " << h_value << std::endl;
        double h_ak = h_low/h_value;
        cudaMemcpy(d_ak, &h_ak, sizeof(double), cudaMemcpyHostToDevice);
        // h_xk = h_xk + ak*h_pk;
        vectorAdd<<<grid, block>>>(d_x, d_p, d_ak, d_x, rows);
        // h_rk = h_rk - ak*Ap;
        vectorMinus<<<grid, block>>>(d_r, d_Ap, d_ak, d_r, rows);
        // up = h_rk.dot(h_rk);
        vectorDotElementWise<<<grid, block>>>(d_r, d_r, d_temp, rows);
        vectorDotValue<<<grid, block>>>(d_temp, d_temp, rows);
        double h_up = 0.0;
        cudaMemcpy(&h_up, d_temp, sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "Res is " << sqrt(h_up) << std::endl;
        if(sqrt(h_up) < 1e-6) break;
        // double bk = up/low;
        double h_bk = h_up/h_low;
        cudaMemcpy(d_bk, &h_bk, sizeof(double), cudaMemcpyHostToDevice);
        // h_pk = h_rk + bk*h_pk;
        vectorAdd<<<grid, block>>>(d_r,d_p,d_bk,d_p,rows);
        // i++;   
        
        // break;
    }

    // 清理
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cudaFree(d_temp);
    cudaFree(d_up);
    cudaFree(d_low);
    cudaFree(d_ak);
    cudaFree(d_bk);
    cudaFree(d_value);
    return 0;
}
