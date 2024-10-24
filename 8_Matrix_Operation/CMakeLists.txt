cmake_minimum_required(VERSION 3.16)

project("main" LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
find_package(Eigen3 REQUIRED)
link_directories(${Eigen_INCLUDE_DIRS})

add_executable(main)

target_sources(main PRIVATE "main.cu")

target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
target_link_libraries(main Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME}
   ${Eigen_LIBRARIES}
   )