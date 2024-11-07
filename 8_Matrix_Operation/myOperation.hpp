#ifndef MYOPERATION_H
#define MYOPERATION_H

#include <iostream>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <math.h>

// Allocate and transfer
template<typename T>
void transferMatrixToCUDA(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& hostMatrix, T*& deviceMatrix, std::size_t rows, std::size_t cols);
 
template<typename T>
void transferVectorToCUDA(const Eigen::Matrix<T, Eigen::Dynamic, 1>& hostMatrix, T*& deviceMatrix, std::size_t rows, bool flag);
 
__global__ void matrixMultiVector(double* d_matrix, double* d_vector_in,double* d_vector_out, int rows, int cols);
 
__global__ void vectorMultiScale(double* d_vector, double s,int rows);

__global__ void vectorAdd(double* d_vector1, double* d_vector2, double* scale, double* d_out, int rows);

__global__ void vectorMinus(double* d_vector1, double* d_vector2, double* scale, double* d_out, int rows);
 
__global__ void vectorDotElementWise(double* d_vector1, double* d_vector2, double* d_vector3, int rows);

__global__ void vectorDotValue(double* d_vector_in, double* d_vector_out, int rows);

__global__ void vectorDotValuePara(double* d_vector_in, double* d_vector_out, int rows);


#endif // MYOPERATION_H