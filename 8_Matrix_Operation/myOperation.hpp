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
 

// template<typename data_type>
// void allocateToCUDA(data_type *&deviceMatrix, int rows, int cols, bool flag);


// template<typename T>
// __global__ void matMultiMat(T* a, T* b);

// template<typename T>
// __global__ void matMultiVec(T* a, T* b);

// template<typename T>
// __global__ void vecAdd(T* a, T* b);

// template<typename T>
// __global__ void vecDot(T* a, T* b);

// template<Typename T, Typename scale>
// __global__ void scaleVector(T* a, scale* s);


#endif // MYOPERATION_H