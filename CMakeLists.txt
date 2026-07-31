cmake_minimum_required(VERSION 3.16)
project(mmx_opencl_plotter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find mmx-node
set(MMX_NODE_DIR $ENV{HOME}/mmx-node CACHE PATH "Path to mmx-node")

include_directories(
    ${OpenCL_INCLUDE_DIRS}
    ${MMX_NODE_DIR}/include
    ${MMX_NODE_DIR}/generated/include
    ${MMX_NODE_DIR}/generated/vm/include
    ${MMX_NODE_DIR}/generated/contract/include
    ${MMX_NODE_DIR}/generated/operation/include
    ${MMX_NODE_DIR}/generated/solution/include
    ${MMX_NODE_DIR}/vnx-base/include
    ${MMX_NODE_DIR}/vnx-base/generated/include
    ${MMX_NODE_DIR}/build_opencl/vnx-base
    ${MMX_NODE_DIR}/vnx-addons/include
    ${MMX_NODE_DIR}/vnx-addons/generated/include
    ${MMX_NODE_DIR}/uint256_t
    ${MMX_NODE_DIR}/uint256_t/uint128_t
    ${MMX_NODE_DIR}/lib/libbech32/libbech32/../include/libbech32
    ${MMX_NODE_DIR}/lib/secp256k1/include
)

link_directories(
    ${MMX_NODE_DIR}/build_opencl
    ${MMX_NODE_DIR}/build_opencl/vnx-base
    ${MMX_NODE_DIR}/build_opencl/vnx-addons
)

add_executable(mmx_opencl_plotter plotter.cpp)

target_link_libraries(mmx_opencl_plotter
    mmx_pos
    mmx_pos_verify
    mmx_iface
    vnx_base
    vnx_addons
    ${OpenCL_LIBRARIES}
    OpenMP::OpenMP_CXX
    pthread
    dl
    stdc++fs
)

# Enable parallel algorithms
target_compile_options(mmx_opencl_plotter PRIVATE -fopenmp -D_GLIBCXX_PARALLEL)
