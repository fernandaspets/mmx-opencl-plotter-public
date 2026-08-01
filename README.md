# MMX OpenCL Plotter

OpenCL plotter for MMX Proof-of-Space on AMD GPUs (gfx1100 / RDNA3).
Hybrid GPU+CPU pipeline: F1 on GPU, F2-F9 on GPU or CPU.

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| F1 (SHA-512 + memory-hard) | ✅ Working | Verified byte-for-byte against CPU reference (100/100 match) |
| Table hash (SHA-512 L‖R) | ✅ Working | Verified against CPU (20/20 match) |
| Flat pipeline (F2-F9) | ✅ Working | 79-95% pass rate on mmx_postool |
| Chunked pipeline (k29+) | ⚠️ In progress | PD (Prover Data) chain has sort-order bug |
| Plot file writing | ✅ Working | Y parks, meta parks, PD parks, X parks |
| RAM disk support | ✅ Working | `--ramdisk DIR` flag |

## Building

```bash
cd opencl-plotter/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Requires: OpenCL (ROCm), mmx-node headers/libs for verification.

## Usage

```bash
# Generate a plot (flat pipeline, k ≤ 26)
LD_LIBRARY_PATH=/opt/rocm/lib ./mmx_opencl_plotter <plot_id_hex> <farmer_key_hex> <output_dir> --k 26

# Chunked pipeline (k29+, uses ~66GB RAM for k29)
LD_LIBRARY_PATH=/opt/rocm/lib ./mmx_opencl_plotter <plot_id_hex> <farmer_key_hex> <output_dir> --k 29 --chunked

# With RAM disk (recommended for speed)
./mmx_opencl_plotter <pid> <fkey> /mnt/hdd/ --ramdisk /mnt/ramdisk --k 26

# Disable GPU display yield (headless)
./mmx_opencl_plotter <pid> <fkey> <outdir> --k 26 --no-yield
```

### Plot ID

Plot ID is a 32-byte hex string (64 hex chars). In the official CUDA plotter, it's
derived as `SHA256("MMX/PLOTID/OG" || ksize || seed || farmer_key)`, but any 32-byte
value works — the Prover reads it from the file header.

## Verification

```bash
# Verify with mmx_postool
LD_LIBRARY_PATH=/opt/rocm/lib ~/mmx-node/build_opencl/tools/mmx_postool -f <plot.plot> -n 20 -v
```

Or use the included tool:
```bash
./tools/verify_plot.sh --k 18
./tools/verify_plot.sh --k 18 --chunked
```

## Testing

```bash
# Build all tests
bash tests/build_test.sh

# F1 comparison: GPU vs CPU (byte-for-byte)
cd tests && LD_LIBRARY_PATH=/opt/rocm/lib ./test_f1_compare

# Table hash comparison: GPU vs CPU
cd tests && LD_LIBRARY_PATH=/opt/rocm/lib ./test_tablehash

# Compare flat vs chunked PD values
./tools/compare.sh --k 18

# PD chain integrity check
python3 tests/test_pd_chain.py flat
python3 tests/test_pd_chain.py chunked
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full architecture documentation.

## Known Bugs

See [docs/PD-BUG-ANALYSIS.md](docs/PD-BUG-ANALYSIS.md) for the chunked pipeline PD bug analysis.

### Flat pipeline intermittent 0%
Certain plot IDs produce 0% pass rate even with the flat pipeline. This appears to
be a PD edge case with duplicate Y values and metadata tie-breaking. ~80-95% of
random plot IDs pass normally.

### Chunked pipeline PD
The chunked pipeline's PD chain is structurally valid (all indices in bounds) but
positions are incorrect due to sort-order inconsistency between per-bucket GPU sort
and global PD sort. See docs/PD-BUG-ANALYSIS.md for details.

## Repository Structure

```
opencl-plotter/
├── plotter.cpp           # Main plotter (flat + chunked pipelines)
├── pos_recompute.cl      # F1 kernel (SHA-512 + memory-hard hash)
├── table_hash.cl         # Table hash kernel (SHA-512 of L‖R metadata)
├── f2_f9.cl              # GPU kernels (scatter, sort, match, eval)
├── simple_sort.cl        # Odd-even sort kernel
├── bucket_store.h         # In-memory bucket store for chunked pipeline
├── CMakeLists.txt
├── docs/
│   ├── ARCHITECTURE.md
│   └── PD-BUG-ANALYSIS.md
├── tools/
│   ├── verify_plot.sh    # Generate + verify any plot
│   └── compare.sh        # Compare flat vs chunked with same plot ID
├── tests/
│   ├── test_f1_compare.cpp    # F1 GPU vs CPU comparison
│   ├── test_tablehash.cpp     # Table hash GPU vs CPU comparison
│   ├── test_memhash_multi.cpp # Memory-hard hash with multiple work-items
│   ├── test_pd_chain.py       # PD chain integrity verification
│   ├── build_test.sh          # Build script for tests
│   └── legacy/                 # Old test files (archive)
└── build/                # Build output (gitignored)
```

## Performance

| K | Pipeline | Time | Notes |
|---|----------|------|-------|
| 18 | Flat | 0.5s | k18 for testing |
| 26 | Flat | 178s | With RAM disk |
| 29 | Chunked | 14.7 min | GPU chunked, no OOM |

GPU: AMD RX 7900 XTX (gfx1100, RDNA3, 24GB VRAM).

## Constants

- `MEM_HASH_ITER = 256` (memory-hard hash iterations, from mmx-node config.h)
- `LOGBUCKETS = 8` (256 first-level buckets)
- `LOGBUCKETS2 = KSIZE - LOGBUCKETS - 9` (sub-buckets, requires k ≥ 18)
- `N_META = 14`, `N_META_OUT = 12`
- `PARK_SIZE_Y = 8192`, `PARK_SIZE_PD = 2048`, `PARK_SIZE_X = 4096`

## License

MIT
