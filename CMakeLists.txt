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
    ${MMX_NODE_PATH}/generated/contract/include
    ${MMX_NODE_PATH}/generated/operation/include
    ${MMX_NODE_PATH}/generated/solution/include
    ${MMX_NODE_PATH}/build/dist/include
    ${MMX_NODE_PATH}/vnx-base/include
    ${MMX_NODE_PATH}/vnx-base/generated/include
    ${MMX_NODE_PATH}/vnx-addons/generated/include
    ${MMX_NODE_PATH}/build/generated/include
    ${MMX_NODE_PATH}/generated/include
    ${MMX_NODE_PATH}/generated/vm/include
    ${MMX_NODE_PATH}/generated/operation/include
    ${MMX_NODE_PATH}/uint256_t
    ${MMX_NODE_PATH}/uint256_t/uint128_t
    ${MMX_NODE_PATH}/lib/secp256k1/include
    ${MMX_NODE_PATH}/lib/libbech32/include
    ${MMX_NODE_PATH}/RNP/include
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

# GPU device source (shared by GPU tests)
set(GPU_SOURCES src/gpu_device.cpp src/gpu_l1_pipeline.cpp)

# F1 kernel test (requires GPU)
add_executable(test_f1 tests/test_f1.cpp ${GPU_SOURCES})
target_link_libraries(test_f1 PRIVATE ${OpenCL_LIBRARIES})
add_test(NAME test_f1 COMMAND test_f1)
set_target_properties(test_f1 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# Table hash kernel test (requires GPU)
add_executable(test_table_hash tests/test_table_hash.cpp ${GPU_SOURCES})
target_link_libraries(test_table_hash PRIVATE ${OpenCL_LIBRARIES})
add_test(NAME test_table_hash COMMAND test_table_hash)
set_target_properties(test_table_hash PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# Full table pipeline test (sort + match + GPU hash vs CPU)
add_executable(test_sha512_gpu tests/test_sha512_gpu.cpp ${GPU_SOURCES})
target_link_libraries(test_sha512_gpu PRIVATE ${OpenCL_LIBRARIES} ssl crypto)
add_test(NAME test_sha512_gpu COMMAND test_sha512_gpu)
set_target_properties(test_sha512_gpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_full_pipeline tests/test_full_pipeline.cpp ${GPU_SOURCES} src/plot_pipeline.cpp)
target_link_libraries(test_full_pipeline PRIVATE ${OpenCL_LIBRARIES} mmx_iface vnx_base_static)
add_test(NAME test_full_pipeline COMMAND test_full_pipeline)
set_target_properties(test_full_pipeline PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

# Plot writer object library (links against mmx-node)
add_library(plot_writer OBJECT src/plot_writer.cpp)
target_link_libraries(plot_writer PUBLIC mmx_iface vnx_base_static)
target_include_directories(plot_writer PUBLIC src ${OpenCL_INCLUDE_DIRS})

add_executable(test_table_pipeline tests/test_table_pipeline.cpp ${GPU_SOURCES})
target_link_libraries(test_table_pipeline PRIVATE ${OpenCL_LIBRARIES} ssl crypto)
add_test(NAME test_table_pipeline COMMAND test_table_pipeline)
set_target_properties(test_table_pipeline PROPERTIES
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

# === MAIN PLOTTER ===
# Build per-ksize binaries matching reference convention
set(PLOTTER_SOURCES src/main.cpp ${GPU_SOURCES} src/plot_pipeline.cpp src/plot_writer.cpp src/gpu_manager.cpp src/multi_pipeline.cpp)
target_link_libraries(plot_writer PUBLIC mmx_iface vnx_base_static)

foreach(KSIZE_VAL 18 20 21 22 24 25 26 27 28 29 30 31 32)
    set(TARGET_NAME mmx_opencl_plot_k${KSIZE_VAL})
    add_executable(${TARGET_NAME} ${PLOTTER_SOURCES})
    target_include_directories(${TARGET_NAME} PRIVATE src)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCL_LIBRARIES} mmx_iface vnx_base)
    target_compile_definitions(${TARGET_NAME} PRIVATE KSIZE=${KSIZE_VAL})
endforeach()
