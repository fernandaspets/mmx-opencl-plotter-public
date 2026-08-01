#!/bin/bash
cd "$(dirname "$0")/.."

MMX_NODE=~/mmx-node
MMX_HOME=~/mmx-data
BUILD="$MMX_NODE/build_opencl"

echo "Building test_f1_compare..."

g++ -std=c++17 -O2 -o tests/test_f1_compare tests/test_f1_compare.cpp \
    -I"$MMX_NODE/include" \
    -I"$MMX_NODE/generated/include" \
    -I"$MMX_NODE/generated/contract/include" \
    -I"$MMX_NODE/generated/operation/include" \
    -I"$MMX_NODE/generated/solution/include" \
    -I"$MMX_NODE/generated/vm/include" \
    -I"$MMX_NODE/vnx-base/include" \
    -I"$MMX_NODE/vnx-base/generated/include" \
    -I"$MMX_NODE/vnx-addons/generated/include" \
    -I"$BUILD/vnx-base" \
    -I"$MMX_HOME/include" \
    -I"$MMX_NODE/uint256_t" \
    -I"$MMX_NODE/uint256_t/uint128_t" \
    "$BUILD/libmmx_pos_verify.a" \
    "$BUILD/libmmx_pos.a" \
    "$BUILD/libuint256_t.a" \
    "$BUILD/libmmx_iface.so" \
    "$BUILD/vnx-base/libvnx_base.so" \
    -L/opt/rocm/lib -lOpenCL \
    -lssl -lcrypto \
    -lpthread \
    -Wl,-rpath,"$BUILD" \
    -Wl,-rpath,"$BUILD/vnx-base" \
    -DCL_TARGET_OPENCL_VERSION=120 \
    2>&1

echo "Build exit: $?"
