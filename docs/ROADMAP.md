# MMX OpenCL Plotter — Production Roadmap

## Current State
- [x] Project structure (src/, kernels/, tests/, golden/)
- [x] plot_config.h — constants matching reference config.h
- [x] plot_params.h — PlotParams struct matching reference input_t
- [x] pid_derive.h — seed gen + PID derivation (OG + NFT)
- [x] plot_codec.h — encode_symbol, decode_symbol, line_point, bit ops
- [x] Tests: test_plot_codec (6 tests), test_pid_derive (8 tests) — 100% pass

## Phase 1: GPU Foundation
- [ ] gpu_device.h/cpp — OpenCL device abstraction (context, queue, kernel loading)
- [ ] kernels/sha512.cl — SHA-512 GPU implementation
- [ ] test_sha512.cpp — SHA-512 GPU vs CPU match
- [ ] kernels/f1.cl — F1 computation (gen_mem_array + mem_hash)
- [ ] test_f1.cpp — F1 GPU vs CPU match (golden vectors)

## Phase 2: Core Pipeline
- [ ] kernels/table_hash.cl — F2-F9 hash kernel (SHA-512 of L||R + metadata)
- [ ] kernels/sort.cl — Radix sort by Y values
- [ ] kernels/match.cl — Match Y with Y+1 pairs
- [ ] table_pipeline.h/cpp — F2-F9 forward pass (one clean pipeline)
- [ ] test_pipeline.cpp — Integration test: full F1-F9 produces correct match counts

## Phase 3: Plot Writing
- [ ] plot_writer.h/cpp — Write plot file (header, Y, meta, PD, X parks)
- [ ] test_plot_writer.cpp — Write plot, verify with mmx_postool
- [ ] test_compression.cpp — C0-C15 all produce valid plots
- [ ] test_ssd.cpp — SSD mode (no metadata table) produces valid plot
- [ ] test_nft.cpp — NFT plot with contract address

## Phase 4: Production CLI
- [ ] cli.h/cpp — Command line matching reference 1:1 (-C, --ssd, -c, -f, -n, -t, -d, etc.)
- [ ] file_copy.h/cpp — Copy to final dirs, remote copy (plot-sink)
- [ ] main.cpp — Entry point with plot loop, signal handling, graceful exit
- [ ] test_cli.cpp — CLI argument parsing tests

## Phase 5: Optimization
- [ ] Warp-parallel F1 (from legacy_reference)
- [ ] GPU-resident M_curr (no PCIe transfer between tables)
- [ ] Phase 2 compaction (mark + remap, 104.6% vs CUDA)
- [ ] Multi-GPU support (--device, -r)
- [ ] Benchmark mode (--benchmark)

## Phase 6: Multi-Machine Testing
- [ ] AMD 7900 XTX (gfx1100) — primary dev machine
- [ ] NVIDIA dual P40 (honeypot) — cross-platform validation
- [ ] 4x RTX 6000 Pro (super) — when available, multi-GPU testing
- [ ] CI script: build + test on all machines
