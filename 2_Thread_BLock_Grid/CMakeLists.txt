cmake_minimum_required(VERSION 3.16) # CMake_最低版本_需求

project("main" LANGUAGES CXX CUDA) # 建立一個CMake專案，並給予專案名稱

add_executable(main) # 建立一個CMake目標，目標為可執行檔

target_sources(main PRIVATE "main.cu") # 指定建置該CMake目標時所使用的來源檔案
