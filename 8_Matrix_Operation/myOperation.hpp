#include <iostream>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <math.h>
#ifndef MYOPERATION_H
#define MYOPERATION_H

template<typename T>
__global__ void matMultiMat(T* a, T* b);

template<typename T>
__global__ void matMultiVec(T* a, T* b);

template<typename T>
__global__ void vecAdd(T* a, T* b);

template<typename T>
__global__ void vecDot(T* a, T* b);


#endif // MYOPERATION_H