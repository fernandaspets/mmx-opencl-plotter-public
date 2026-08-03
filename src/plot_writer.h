#ifndef MMX_PLOT_WRITER_H
#define MMX_PLOT_WRITER_H

#include "plot_config.h"
#include "plot_codec.h"
#include "plot_pipeline.h"
#include <mmx/PlotHeader.hxx>
#include <vector>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

namespace mmx {

// Plot file encoding constants
constexpr int MAX_LPX2SIZE = 63;   // max bits for 2 * (ksize - 0) - 1 = 2*32-1 = 63

// Compute bits per X pair from ksize and compression level (xbits = C)
// Matches CUDA: X2SIZE = 2 * XBITS - 1 where XBITS = KSIZE - C
inline uint32_t compute_x2size(uint32_t ksize, uint32_t xbits) {
    uint32_t XBITS = ksize - xbits;
    return 2 * XBITS - 1;
}

// Plot writer: fills PlotHeader + encodes parks + writes file
class PlotWriter {
public:
    PlotWriter(uint32_t ksize, uint32_t xbits = 0, bool has_meta = true)
        : ksize(ksize), xbits(xbits), x2size(compute_x2size(ksize, xbits)), has_meta(has_meta) {}

    // Write a complete plot file given the pipeline results
    void write_file(
        const std::string& file_path,
        const uint8_t* plot_id_32,
        const uint8_t* farmer_key_33,
        const PlotData& data,
        const uint8_t* contract_32 = nullptr);  // optional contract for NFT

private:
    uint32_t ksize;
    uint32_t xbits;
    uint32_t x2size;  // computed: 2 * (ksize - xbits) - 1
    bool has_meta;

    // Compute header fields
    void compute_header(std::shared_ptr<PlotHeader>& header, const PlotData& data);

    // Encode Y park (write deltas between consecutive Y values)
    void encode_y_park(std::vector<uint8_t>& park, const uint32_t* Y, size_t count,
                       uint32_t max_bytes);

    // Encode PD park
    void encode_pd_park(std::vector<uint8_t>& park,
                        const std::vector<PDEntry>& entries,
                        uint32_t max_bytes);

    // Encode metadata park
    void encode_meta_park(std::vector<uint8_t>& park,
                          const std::vector<PlotEntry>& entries,
                          size_t start, size_t count,
                          uint32_t max_bytes);

    // Encode X pairs park (for table 2)
    void encode_x_park(std::vector<uint8_t>& park,
                       const std::vector<uint32_t>& x_pairs,
                       size_t start, size_t count,
                       uint32_t max_bytes);
};

} // namespace mmx

#endif // MMX_PLOT_WRITER_H
