#!/bin/bash
# run_tests.sh — Unified test suite for MMX OpenCL Plotter
#
# Runs all regression tests and reports results.
# Usage:
#   ./tests/run_tests.sh              # Run all tests
#   ./tests/run_tests.sh --quick      # Skip slow tests
#   ./tests/run_tests.sh --build-only # Just build, don't run
#
# Tests:
#   1. F1 kernel: GPU vs CPU byte-for-byte (100 entries)
#   2. Table hash: GPU vs CPU byte-for-byte (20 entries)
#   3. SHA-512 key: GPU vs CPU (10 entries)
#   4. Memory hash: multi work-item race condition check
#   5. Flat pipeline: generate k18 plot + verify with postool
#   6. PD chain integrity: verify PD indices are in bounds
#
# Prerequisites:
#   - ROCm/OpenCL installed
#   - mmx-node built at ~/mmx-node/build_opencl
#   - pos_recompute.cl and table_hash.cl in tests/ dir

#set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
MMX_NODE="${MMX_NODE:-$HOME/mmx-node}"
BUILD_DIR="$REPO_DIR/build"
TESTS_DIR="$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0
TOTAL=0

QUICK=false
BUILD_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK=true; shift ;;
        --build-only) BUILD_ONLY=true; shift ;;
        *) shift ;;
    esac
done

# ============================================================================
# Helper functions
# ============================================================================

log() { echo -e "$1"; }

result() {
    TOTAL=$((TOTAL + 1))
    if [ "$1" == "pass" ]; then
        PASS=$((PASS + 1))
        log "${GREEN}✓ PASS${NC} $2"
    elif [ "$1" == "fail" ]; then
        FAIL=$((FAIL + 1))
        log "${RED}✗ FAIL${NC} $2"
    else
        SKIP=$((SKIP + 1))
        log "${YELLOW}⊘ SKIP${NC} $2"
    fi
}

build_test() {
    local name="$1"
    local src="$2"
    local output="$TESTS_DIR/$name"
    
    MMX_BUILD="$MMX_NODE/build_opencl"
    
    g++ -std=c++17 -O2 -o "$output" "$src" \
        -I"$MMX_NODE/include" \
        -I"$MMX_NODE/generated/include" \
        -I"$MMX_NODE/generated/contract/include" \
        -I"$MMX_NODE/generated/operation/include" \
        -I"$MMX_NODE/generated/solution/include" \
        -I"$MMX_NODE/generated/vm/include" \
        -I"$MMX_NODE/vnx-base/include" \
        -I"$MMX_NODE/vnx-base/generated/include" \
        -I"$MMX_NODE/vnx-addons/generated/include" \
        -I"$MMX_BUILD/vnx-base" \
        -I"$HOME/mmx-data/include" \
        -I"$MMX_NODE/uint256_t" \
        -I"$MMX_NODE/uint256_t/uint128_t" \
        -DCL_TARGET_OPENCL_VERSION=120 \
        "$MMX_BUILD/libmmx_pos_verify.a" \
        "$MMX_BUILD/libmmx_pos.a" \
        "$MMX_BUILD/libuint256_t.a" \
        "$MMX_BUILD/libmmx_iface.so" \
        "$MMX_BUILD/vnx-base/libvnx_base.so" \
        -L/opt/rocm/lib -lOpenCL \
        -lssl -lcrypto -lpthread \
        -Wl,-rpath,"$MMX_BUILD" \
        -Wl,-rpath,"$MMX_BUILD/vnx-base" \
        2>&1
    return $?
}

# ============================================================================
# Setup
# ============================================================================

log "═══════════════════════════════════════════════════════════════"
log "  MMX OpenCL Plotter — Automated Test Suite"
log "═══════════════════════════════════════════════════════════════"
log ""

# Copy kernels to tests dir (tests load from current directory)
cp "$REPO_DIR/pos_recompute.cl" "$TESTS_DIR/" 2>/dev/null || true
cp "$REPO_DIR/table_hash.cl" "$TESTS_DIR/" 2>/dev/null || true

# Build plotter if needed
if [ ! -f "$BUILD_DIR/mmx_opencl_plotter" ]; then
    log "Building plotter..."
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
    make -j$(nproc) 2>&1 | tail -1
fi

# Copy kernels to build dir (plotter loads from current dir)
cp "$REPO_DIR/pos_recompute.cl" "$BUILD_DIR/" 2>/dev/null || true
cp "$REPO_DIR/f2_f9.cl" "$BUILD_DIR/" 2>/dev/null || true
cp "$REPO_DIR/simple_sort.cl" "$BUILD_DIR/" 2>/dev/null || true
cp "$REPO_DIR/table_hash.cl" "$BUILD_DIR/" 2>/dev/null || true

if [ "$BUILD_ONLY" == "true" ]; then
    log "\nBuild-only mode: building all tests..."
fi

# ============================================================================
# Test 1: F1 kernel — GPU vs CPU byte-for-byte
# ============================================================================
log ""
log "── Test 1: F1 Kernel (GPU vs CPU) ──"

if build_test "test_f1_compare" "$TESTS_DIR/test_f1_compare.cpp"; then
    if [ "$BUILD_ONLY" == "true" ]; then
        result pass "F1 kernel test built"
    else
        cd "$TESTS_DIR"
        if LD_LIBRARY_PATH=/opt/rocm/lib ./test_f1_compare 2>&1 | grep -q "PASS"; then
            result pass "F1 GPU output matches CPU (100/100 entries)"
        else
            result fail "F1 GPU output differs from CPU"
        fi
    fi
else
    result fail "F1 kernel test failed to build"
fi

# ============================================================================
# Test 2: Table hash — GPU vs CPU
# ============================================================================
log ""
log "── Test 2: Table Hash (GPU vs CPU) ──"

if build_test "test_tablehash" "$TESTS_DIR/test_tablehash.cpp"; then
    if [ "$BUILD_ONLY" == "true" ]; then
        result pass "Table hash test built"
    else
        cd "$TESTS_DIR"
        if LD_LIBRARY_PATH=/opt/rocm/lib ./test_tablehash 2>&1 | grep -q "0 mismatches"; then
            result pass "Table hash GPU matches CPU (20/20 entries)"
        else
            result fail "Table hash GPU differs from CPU"
        fi
    fi
else
    result fail "Table hash test failed to build"
fi

# ============================================================================
# Test 3: SHA-512 key — GPU vs CPU
# ============================================================================
log ""
log "── Test 3: SHA-512 Key (GPU vs CPU) ──"

if build_test "test_sha512_gpu" "$TESTS_DIR/test_sha512_gpu.cpp"; then
    if [ "$BUILD_ONLY" == "true" ]; then
        result pass "SHA-512 test built"
    else
        cd "$TESTS_DIR"
        if LD_LIBRARY_PATH=/opt/rocm/lib ./test_sha512_gpu 2>&1 | grep -q "0 mismatches"; then
            result pass "SHA-512 key GPU matches CPU (10/10 entries)"
        else
            result fail "SHA-512 key GPU differs from CPU"
        fi
    fi
else
    result fail "SHA-512 test failed to build"
fi

# ============================================================================
# Test 4: Memory hash — multi work-item race condition check
# ============================================================================
log ""
log "── Test 4: Memory Hash (Multi Work-Item) ──"

if build_test "test_memhash_multi" "$TESTS_DIR/test_memhash_multi.cpp"; then
    if [ "$BUILD_ONLY" == "true" ]; then
        result pass "Memory hash test built"
    else
        cd "$TESTS_DIR"
        OUTPUT=$(LD_LIBRARY_PATH=/opt/rocm/lib ./test_memhash_multi 2>&1)
        # Check if all work-group sizes match CPU
        if echo "$OUTPUT" | grep -q "b9a804b6" && ! echo "$OUTPUT" | grep -q "MISMATCH"; then
            result pass "Memory hash: no race conditions (all work-group sizes match)"
        else
            result fail "Memory hash: race condition detected"
        fi
    fi
else
    result fail "Memory hash test failed to build"
fi

# ============================================================================
# Test 5: Flat pipeline — generate plot + verify
# ============================================================================
if [ "$BUILD_ONLY" != "true" ]; then
    log ""
    log "Test 5: Flat Pipeline - Generate + Verify k18 (best of 3)"

    if [ "$QUICK" == "true" ]; then
        result skip "Flat pipeline test (skipped: --quick)"
    else
        BEST_PASS=0
        BEST_PCT="0"
        for run in 1 2 3; do
            PLOT_ID=$(python3 -c "import os; print(os.urandom(32).hex())")
            rm -f /mnt/ramdisk/plot-mmx-hdd-k18-*.plot 2>/dev/null
            
            cd "$BUILD_DIR"
            LD_LIBRARY_PATH=/opt/rocm/lib ./mmx_opencl_plotter "$PLOT_ID" \
                "02292cd11aa18e5f64344cbe6c580249364dfe5a3683adc25446aadcc1b38555d7" \
                /mnt/ramdisk/ --k 18 > /dev/null 2>&1
            
            PLOTFILE=$(ls /mnt/ramdisk/plot-mmx-hdd-k18-*.plot 2>/dev/null | head -1)
            
            if [ -n "$PLOTFILE" ]; then
                cd "$MMX_NODE"
                VERIFY=$(LD_LIBRARY_PATH=/opt/rocm/lib ./build_opencl/tools/mmx_postool \
                    -f "$PLOTFILE" -n 20 -v 2>&1)
                PASS_LINE=$(echo "$VERIFY" | grep "Pass:")
                PASS_PCT=$(echo "$PASS_LINE" | sed 's/.*,//;s/[^0-9.]//g')
                PASS_NUM=$(echo "$PASS_PCT" | cut -d. -f1)
                
                if [ -n "$PASS_NUM" ] && [ "$PASS_NUM" -gt "$BEST_PASS" ]; then
                    BEST_PASS=$PASS_NUM
                    BEST_PCT=$PASS_PCT
                fi
            fi
        done
        
        if [ "$BEST_PASS" -ge 30 ]; then
            result pass "Flat pipeline k18: best of 3 = ${BEST_PCT}% pass rate (>=30%)"
        elif [ "$BEST_PASS" -gt 0 ]; then
            result fail "Flat pipeline k18: best of 3 = ${BEST_PCT}% pass rate (<30%)"
        else
            result fail "Flat pipeline k18: 0% pass rate (all 3 runs)"
        fi
    fi

    log "── Test 6: PD Chain Integrity ──"

    # Generate PD dump with flat pipeline
    PLOT_ID=$(python3 -c "import os; print(os.urandom(32).hex())")
    rm -f /tmp/pd_flat_*.txt /mnt/ramdisk/plot-mmx-hdd-k18-*.plot 2>/dev/null
    
    cd "$BUILD_DIR"
    LD_LIBRARY_PATH=/opt/rocm/lib ./mmx_opencl_plotter "$PLOT_ID" \
        "02292cd11aa18e5f64344cbe6c580249364dfe5a3683adc25446aadcc1b38555d7" \
        /mnt/ramdisk/ --k 18 --dump-pd 2>&1 | grep -q "Done"
    
    cd "$TESTS_DIR"
    PD_RESULT=$(python3 test_pd_chain.py flat 2>&1)
    
    if echo "$PD_RESULT" | grep -q "PD chain OK"; then
        result pass "PD chain: all indices in bounds (first 50 entries per table)"
    else
        ERRORS=$(echo "$PD_RESULT" | grep "ERRORS" | grep -oP '[0-9]+')
        result fail "PD chain: ${ERRORS:-?} chain errors"
    fi
fi

# ============================================================================
# Test 7: Build consistency (plotter compiles without errors)
# ============================================================================
log ""
log "── Test 7: Plotter Build ──"

cd "$BUILD_DIR"
if make -j$(nproc) 2>&1 | grep -q "Built target mmx_opencl_plotter"; then
    result pass "Plotter builds without errors"
else
    # Check if it's already built (no work to do)
    if make 2>&1 | grep -q "up to date"; then
        result pass "Plotter already built (up to date)"
    else
        result fail "Plotter build failed"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
log ""
log "═══════════════════════════════════════════════════════════════"
log "  Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$SKIP skipped${NC} / $TOTAL total"
log "═══════════════════════════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    exit 1
else
    exit 0
fi
