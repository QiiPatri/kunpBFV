# 设置cmake的最低版本号为3.26
cmake_minimum_required(VERSION 3.26)

project(modules_cuda CUDA)

# 设置项目的C++标准为C++20
set(CMAKE_CUDA_STANDARD 20)

set(CUDA_NVCC_FLAGS --expt-relaxed-constexpr)

add_executable(bfv_demo bfv_demo.cu)

# 设置可执行程序的target属性
set_target_properties(bfv_demo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# 链接NTL库
target_link_libraries(bfv_demo ntl)

