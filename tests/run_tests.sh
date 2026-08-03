#!/bin/bash
# Run all unit tests (no GPU required)
cd "$(dirname "$0")/.."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
make -j$(nproc) 2>&1 | tail -5
echo ""
echo "=== Running Tests ==="
ctest --output-on-failure
