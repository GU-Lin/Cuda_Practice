cmake_minimum_required(VERSION 3.16)

project("main" LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
add_executable(main)

target_sources(main PRIVATE "main.cu")

target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)