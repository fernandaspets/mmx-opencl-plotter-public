cmake_minimum_required(VERSION 3.20)
project(mmx_opencl_plotter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCL
find_package(OpenCL REQUIRED)

# MMX node paths
set(MMX_NODE_PATH "${CMAKE_SOURCE_DIR}/../../mmx-node" CACHE PATH "Path to mmx-node")
if(NOT EXISTS "${MMX_NODE_PATH}/CMakeLists.txt")
    set(MMX_NODE_PATH "${CMAKE_SOURCE_DIR}/../mmx-node" CACHE PATH "Path to mmx-node" FORCE)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCL_INCLUDE_DIRS}
    ${MMX_NODE_PATH}/include
    ${MMX_NODE_PATH}/generated/include
    ${MMX_NODE_PATH}/build/dist/include
    ${MMX_NODE_PATH}/vnx-base/include
    ${MMX_NODE_PATH}/build/generated/include
    ${MMX_NODE_PATH}/generated/include
    ${MMX_NODE_PATH}/generated/vm/include
    ${MMX_NODE_PATH}/generated/operation/include
    ${MMX_NODE_PATH}/uint256_t
    ${MMX_NODE_PATH}/uint256_t/uint128_t
    ${MMX_NODE_PATH}/lib/secp256k1/include
)

# Link directories
link_directories(
    ${MMX_NODE_PATH}/build
    ${MMX_NODE_PATH}/build/dist/lib
    ${MMX_NODE_PATH}/build/lib/libbech32/libbech32
    ${MMX_NODE_PATH}/build/vnx-base
)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O2 -Wall -Wno-unused-parameter -Wno-unused-but-set-variable)
endif()

# OpenMP (optional, for CPU parallelism)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
    add_link_options(${OpenMP_CXX_FLAGS})
endif()

# === TESTS ===
enable_testing()

# Standalone test (no MMX deps)
add_executable(test_plot_codec tests/test_plot_codec.cpp)
add_test(NAME test_plot_codec COMMAND test_plot_codec)
set_target_properties(test_plot_codec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# Test with MMX deps
add_executable(test_pid_derive tests/test_pid_derive.cpp)
target_link_libraries(test_pid_derive PRIVATE
    ${OpenCL_LIBRARIES}
    mmx_iface vnx_base_static bech32 uint256_t
)
add_test(NAME test_pid_derive COMMAND test_pid_derive)
set_target_properties(test_pid_derive PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# === MAIN PLOTTER (to be implemented) ===
# add_executable(mmx_opencl_plotter src/main.cpp ...)
