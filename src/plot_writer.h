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

// Plot file encoding constants (used in park encoding)
constexpr int LPX2SIZE = 36;       // bits per X value in stored format

// Plot writer: fills PlotHeader + encodes parks + writes file
class PlotWriter {
public:
    PlotWriter(uint32_t ksize, uint32_t xbits = 0, bool has_meta = true)
        : ksize(ksize), xbits(xbits), has_meta(has_meta) {}

    // Write a complete plot file given the pipeline results
    void write_file(
        const std::string& file_path,
        const uint8_t* plot_id_32,
        const uint8_t* farmer_key_33,
        const PlotData& data);

private:
    uint32_t ksize;
    uint32_t xbits;
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
