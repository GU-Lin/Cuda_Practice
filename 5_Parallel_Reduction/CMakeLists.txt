cmake_minimum_required(VERSION 3.16)

project("main" LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ")

add_executable(main)

target_sources(main PRIVATE "main.cu")

