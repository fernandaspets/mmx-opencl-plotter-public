#ifndef MMX_PLOT_WRITER_H
#define MMX_PLOT_WRITER_H

#include "plot_config.h"
#include "plot_codec.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

namespace mmx {

// Plot file format constants
constexpr int PARK_SIZE_X = 2048;
constexpr int PARK_SIZE_Y = 2048;
constexpr int PARK_SIZE_PD = 2048;
constexpr int PARK_SIZE_META = 2048;
constexpr int LPX2SIZE = 36;  // bits per X value in stored format

// Plot header information (compatible with MMX node format)
struct PlotHeader {
    uint32_t version = 0;
    uint32_t ksize = 26;
    uint32_t xbits = 0;  // 0 = C0 (no compression)
    bool has_meta = true;  // HDD mode

    // Park sizes
    uint32_t park_size_x = PARK_SIZE_X;
    uint32_t park_size_y = PARK_SIZE_Y;
    uint32_t park_size_pd = PARK_SIZE_PD;
    uint32_t park_size_meta = PARK_SIZE_META;

    // Maximum bytes per park (computed from sizes)
    uint32_t park_bytes_x = 0;
    uint32_t park_bytes_y = 0;
    uint32_t park_bytes_pd = 0;
    uint32_t park_bytes_meta = 0;

    uint32_t entry_bits_x = LPX2SIZE;
    uint32_t num_entries_y = 0;

    // Offsets in file (filled during write)
    uint64_t table_offset_y = 0;     // Y table start
    uint64_t table_offset_meta = 0;     // metadata table start
    uint64_t table_offset_x = 0;        // X pairs table start
    uint64_t plot_size = 0;             // total file size

    // PD table offsets (7 tables: F9→F8, F8→F7, ..., F3→F2)
    std::vector<uint64_t> table_offset_pd = std::vector<uint64_t>(7, 0);

    // Binary plot id and farmer key (32 bytes + 33 bytes)
    std::vector<uint8_t> plot_id = std::vector<uint8_t>(32, 0);
    std::vector<uint8_t> farmer_key = std::vector<uint8_t>(33, 0);
    // Seed (32 bytes \xe2\x80\x94 unused in current protocol)
    std::vector<uint8_t> seed = std::vector<uint8_t>(32, 0);
};

// Collected data for one table during the pipeline
struct TableData {
    // Matched pairs: (sorted_pos_L_in_prev, sorted_pos_R_in_prev)
    std::vector<std::pair<uint32_t, uint32_t>> matches;

    // PD entries: (pos_in_prev_table, delta_info)
    // For each entry in this table, one PD entry referencing back
    std::vector<std::pair<uint32_t, uint16_t>> pd_entries;

    // For table 2: X pairs (original X values for proof reconstruction)
    std::vector<uint32_t> x_pairs;
};

// Plot writer
class PlotWriter {
public:
    PlotWriter(uint32_t ksize, uint32_t xbits = 0, bool has_meta = true);

    // Add table data from one pipeline iteration
    void add_table_data(int table_idx, const std::vector<TableData>& buckets);

    // Write plot file
    void write_file(const std::string& path,
                    const uint8_t* plot_id_32,
                    const uint8_t* farmer_key_33,
                    const std::vector<uint32_t>& final_Y,
                    const std::vector<std::vector<TableData>>& table_data);

    // Compute PD park encoding
    static std::vector<uint64_t> encode_pd_park(
        const std::vector<std::pair<uint32_t, uint16_t>>& entries,
        uint64_t& total_bits);

private:
    uint32_t ksize;
    uint32_t xbits;
    bool has_meta;

    void compute_plot_header_sizes(PlotHeader& header) const;
};

} // namespace mmx

#endif // MMX_PLOT_WRITER_H
