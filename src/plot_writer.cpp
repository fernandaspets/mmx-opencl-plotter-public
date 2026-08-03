#include "plot_writer.h"
#include <mmx/PlotHeader.hxx>
#include <vnx/Output.h>
#include <fstream>
#include <algorithm>
#include <omp.h>

namespace mmx {

void PlotWriter::compute_header(
    std::shared_ptr<PlotHeader>& header,
    const PlotData& data)
{
    const uint64_t num_entries = data.final_Y.size();

    header->version = 0;
    header->ksize = ksize;
    header->xbits = xbits;
    header->has_meta = has_meta;
    header->seed = hash_t();
    header->park_size_x = PARK_SIZE_X;
    header->park_size_y = PARK_SIZE_Y;
    header->park_size_pd = PARK_SIZE_PD;
    header->park_size_meta = PARK_SIZE_META;

    // Maximum bytes per park
    header->park_bytes_x = cdiv(PARK_SIZE_X * LPX2SIZE, 8);
    header->park_bytes_meta = cdiv(PARK_SIZE_META * ksize * N_META_OUT, 8);
    header->park_bytes_y = 4 + cdiv((PARK_SIZE_Y - 1) * (uint64_t)MAX_AVG_YDELTA_BITS, 8);
    header->park_bytes_pd = cdiv(PARK_SIZE_PD * ksize, 8)
                          + cdiv(PARK_SIZE_PD * (uint64_t)MAX_AVG_OFFSET_BITS, 8);

    header->entry_bits_x = LPX2SIZE;
    header->num_entries_y = num_entries;

    // Table offsets
    header->table_offset_pd.resize(7);
    header->table_offset_y = 0;
    header->table_offset_meta = 0;
    header->table_offset_x = 0;
    header->plot_size = 0;

    // Compute layout
    uint64_t offset = 0;  // header size will be added later after measuring

    header->table_offset_y = offset;
    const uint64_t num_parks_y = cdiv(num_entries, (uint64_t)PARK_SIZE_Y);
    offset += num_parks_y * header->park_bytes_y;

    if(has_meta) {
        header->table_offset_meta = offset;
        const uint64_t num_parks_meta = cdiv(num_entries, (uint64_t)PARK_SIZE_META);
        offset += num_parks_meta * header->park_bytes_meta;
    }

    // PD tables (reverse order: 9→8, 8→7, ..., 3→2)
    // table_offset_pd[0] = PD for table 9→8 (num_entries[9] entries)
    // table_offset_pd[6] = PD for table 3→2 (num_entries[3] entries)
    for(int t = 0; t < 7 && t < (int)data.pd_data.size(); t++) {
        header->table_offset_pd[t] = offset;
        int pd_table_idx = (int)data.pd_data.size() - 1 - t;  // last PD table first
        if(pd_table_idx >= 0 && pd_table_idx < (int)data.pd_data.size()) {
            const uint64_t num_pd = data.pd_data[pd_table_idx].size();
            const uint64_t num_parks_pd = cdiv(num_pd, (uint64_t)PARK_SIZE_PD);
            offset += num_parks_pd * header->park_bytes_pd;
        }
    }

    // X pairs table (table 2 entries)
    header->table_offset_x = offset;
    if(!data.x_pairs.empty()) {
        const uint64_t num_x = data.x_pairs.size() / 2;  // each pair = 2 uint32s
        const uint64_t num_parks_x = cdiv(num_x, (uint64_t)PARK_SIZE_X);
        offset += num_parks_x * header->park_bytes_x;
    }

    header->plot_size = offset;
}

void PlotWriter::write_file(
    const std::string& file_path,
    const uint8_t* plot_id_32,
    const uint8_t* farmer_key_33,
    const PlotData& data)
{
    std::cout << "[Plot] Writing " << data.final_Y.size() << " entries to " << file_path << std::endl;

    // Create header
    auto header = PlotHeader::create();
    std::memcpy(header->plot_id.data(), plot_id_32, 32);
    std::memcpy(header->farmer_key.data(), farmer_key_33, 33);

    compute_header(header, data);

    // Save plot_id/farmer_key using proper constructors
    std::array<uint8_t, 32> pid_arr;
    std::memcpy(pid_arr.data(), plot_id_32, 32);
    header->plot_id = hash_t(pid_arr);
    
    std::vector<uint8_t> fk_vec(farmer_key_33, farmer_key_33 + 33);
    header->farmer_key = pubkey_t(fk_vec);

    // First write: measure header size
    {
        vnx::write_to_file(file_path, header);
    }

    // Get header size
    std::ifstream tmp(file_path, std::ios::binary | std::ios::ate);
    uint64_t header_size = tmp.tellg();
    tmp.close();

    // Align to 4096 (like CUDA plotter)
    header_size = (header_size + 4095) & ~4095;

    // Recompute offsets with header_size
    header->table_offset_y += header_size;
    if(has_meta) header->table_offset_meta += header_size;
    for(int t = 0; t < 7; t++) header->table_offset_pd[t] += header_size;
    header->table_offset_x += header_size;
    header->plot_size += header_size;

    // Second write: header with correct offsets (same binary size)
    vnx::write_to_file(file_path, header);

    // Pad header to 4096 alignment
    {
        std::fstream f(file_path, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(0, std::ios::end);
        uint64_t current_size = f.tellp();
        if(current_size < header_size) {
            std::vector<char> zeros(header_size - current_size, 0);
            f.write(zeros.data(), zeros.size());
        }
    }

    std::cout << "[Plot] Header size: " << header_size << std::endl;
    std::cout << "[Plot] Y offset: " << header->table_offset_y
              << " X offset: " << header->table_offset_x
              << " Total: " << header->plot_size / 1e6 << " MB" << std::endl;

    // Open file for appending parks
    std::ofstream out(file_path, std::ios::binary | std::ios::app);
    if(!out.good()) throw std::runtime_error("Cannot open " + file_path);

    const uint64_t num_entries = data.final_Y.size();

    // Write Y table
    {
        const uint64_t num_parks = cdiv(num_entries, (uint64_t)PARK_SIZE_Y);
        std::cout << "[Plot] Writing Y table (" << num_parks << " parks)..." << std::endl;

        for(uint64_t p = 0; p < num_parks; p++) {
            uint64_t start = p * PARK_SIZE_Y;
            uint64_t count = std::min((uint64_t)PARK_SIZE_Y, num_entries - start);
            std::vector<uint8_t> park(header->park_bytes_y, 0);
            encode_y_park(park, &data.final_Y[start], count, header->park_bytes_y);
            out.write((char*)park.data(), header->park_bytes_y);
        }
    }

    // Write metadata table (HDD mode only)
    if(has_meta && !data.table_entries.empty()) {
        const int last_table = data.table_entries.size() - 1;
        const auto& entries = data.table_entries[last_table];
        const uint64_t num_parks = cdiv(num_entries, (uint64_t)PARK_SIZE_META);
        std::cout << "[Plot] Writing metadata table (" << num_parks << " parks)..." << std::endl;

        for(uint64_t p = 0; p < num_parks; p++) {
            uint64_t start = p * PARK_SIZE_META;
            uint64_t count = std::min((uint64_t)PARK_SIZE_META, num_entries - start);
            std::vector<uint8_t> park(header->park_bytes_meta, 0);
            encode_meta_park(park, entries, start, count, header->park_bytes_meta);
            out.write((char*)park.data(), header->park_bytes_meta);
        }
    }

    // Write PD tables
    for(int t = 0; t < 7 && t < (int)data.pd_data.size(); t++) {
        int pd_idx = (int)data.pd_data.size() - 1 - t;
        const auto& pd = data.pd_data[pd_idx];
        const uint64_t num_parks = cdiv(pd.size(), (uint64_t)PARK_SIZE_PD);
        std::cout << "[Plot] Writing PD table " << pd_idx << " (" << num_parks << " parks)..." << std::endl;

        for(uint64_t p = 0; p < num_parks; p++) {
            uint64_t start = p * PARK_SIZE_PD;
            uint64_t count = std::min((uint64_t)PARK_SIZE_PD, (uint64_t)pd.size() - start);
            // Build PDEntry sub-vector
            std::vector<PDEntry> pd_sub(pd.begin() + start, pd.begin() + start + count);
            std::vector<uint8_t> park(header->park_bytes_pd, 0);
            encode_pd_park(park, pd_sub, header->park_bytes_pd);
            out.write((char*)park.data(), header->park_bytes_pd);
        }
    }

    // Write X pairs table
    if(!data.x_pairs.empty()) {
        const uint64_t num_x_entries = data.x_pairs.size() / 2;
        const uint64_t num_parks = cdiv(num_x_entries, (uint64_t)PARK_SIZE_X);
        std::cout << "[Plot] Writing X pairs table (" << num_parks << " parks)..." << std::endl;

        for(uint64_t p = 0; p < num_parks; p++) {
            uint64_t start = p * PARK_SIZE_X;
            uint64_t count = std::min((uint64_t)PARK_SIZE_X, num_x_entries - start);
            std::vector<uint8_t> park(header->park_bytes_x, 0);
            encode_x_park(park, data.x_pairs, start, count, header->park_bytes_x);
            out.write((char*)park.data(), header->park_bytes_x);
        }
    }

    std::cout << "[Plot] Done writing " << file_path << std::endl;
}

void PlotWriter::encode_y_park(std::vector<uint8_t>& park, const uint32_t* Y, size_t count,
                                uint32_t max_bytes)
{
    std::vector<uint64_t> bit_buf(max_bytes * 2 / sizeof(uint64_t) + 4, 0);

    // First Y value: KSIZE bits
    write_bits(bit_buf, Y[0], 0, ksize);

    uint64_t bit_offset = 32;  // start at byte 4 (after 4-byte header area)
    // Actually the first Y is just at bit 0 with ksize bits
    // Let me redo: bit 0 = first Y value

    // Reset — write first Y at bit 0
    bit_offset = ksize;

    // Y deltas
    for(size_t i = 1; i < count; i++) {
        uint32_t delta = Y[i] - Y[i-1];
        auto [bits, nbits] = encode_symbol((uint8_t)delta);
        write_bits(bit_buf, bits, bit_offset, nbits);
        bit_offset += nbits;
    }

    // Copy to byte buffer
    uint64_t byte_count = (bit_offset + 7) / 8;
    if(byte_count > max_bytes) {
        throw std::runtime_error("Y park overflow: " + std::to_string(byte_count)
            + " > " + std::to_string(max_bytes));
    }
    std::memcpy(park.data(), bit_buf.data(), std::min(byte_count, (uint64_t)max_bytes));
}

void PlotWriter::encode_meta_park(std::vector<uint8_t>& park,
                                   const std::vector<PlotEntry>& entries,
                                   size_t start, size_t count,
                                   uint32_t max_bytes)
{
    std::vector<uint64_t> bit_buf(max_bytes * 2 / sizeof(uint64_t) + 4, 0);
    uint64_t bit_offset = 0;

    for(size_t i = start; i < start + count && i < entries.size(); i++) {
        for(int j = 0; j < N_META; j++) {
            write_bits(bit_buf, entries[i].M[j], bit_offset, ksize);
            bit_offset += ksize;
        }
    }

    uint64_t byte_count = (bit_offset + 7) / 8;
    if(byte_count > max_bytes) {
        throw std::runtime_error("Meta park overflow: " + std::to_string(byte_count)
            + " > " + std::to_string(max_bytes));
    }
    std::memcpy(park.data(), bit_buf.data(), std::min(byte_count, (uint64_t)max_bytes));
}

void PlotWriter::encode_pd_park(std::vector<uint8_t>& park,
                                 const std::vector<PDEntry>& entries,
                                 uint32_t max_bytes)
{
    std::vector<uint64_t> bit_buf(max_bytes * 2 / sizeof(uint64_t) + 4, 0);
    uint64_t bit_offset = 0;

    for(const auto& [pos, delta] : entries) {
        write_bits(bit_buf, pos, bit_offset, ksize);
        bit_offset += ksize;
        write_bits(bit_buf, delta, bit_offset, 16);
        bit_offset += 16;
    }

    uint64_t byte_count = (bit_offset + 7) / 8;
    if(byte_count > max_bytes) {
        throw std::runtime_error("PD park overflow: " + std::to_string(byte_count)
            + " > " + std::to_string(max_bytes));
    }
    std::memcpy(park.data(), bit_buf.data(), std::min(byte_count, (uint64_t)max_bytes));
}

void PlotWriter::encode_x_park(std::vector<uint8_t>& park,
                                const std::vector<uint32_t>& x_pairs,
                                size_t start, size_t count,
                                uint32_t max_bytes)
{
    std::vector<uint64_t> bit_buf(max_bytes * 2 / sizeof(uint64_t) + 4, 0);
    uint64_t bit_offset = 0;

    for(size_t i = start * 2; i < (start + count) * 2 && i < x_pairs.size(); i += 2) {
        write_bits(bit_buf, x_pairs[i], bit_offset, LPX2SIZE);
        bit_offset += LPX2SIZE;
        write_bits(bit_buf, x_pairs[i+1], bit_offset, LPX2SIZE);
        bit_offset += LPX2SIZE;
    }

    uint64_t byte_count = (bit_offset + 7) / 8;
    if(byte_count > max_bytes) {
        throw std::runtime_error("X park overflow: " + std::to_string(byte_count)
            + " > " + std::to_string(max_bytes));
    }
    std::memcpy(park.data(), bit_buf.data(), std::min(byte_count, (uint64_t)max_bytes));
}

} // namespace mmx
