cmake_minimum_required(VERSION 3.16)

project("main" LANGUAGES CXX CUDA)

add_executable(main)

target_sources(main PRIVATE "main.cu")
