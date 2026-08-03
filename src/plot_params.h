#pragma once

#include <cstdint>
#include <string>
#include <array>
#include <vector>
#include <optional>
#include <mmx/PlotHeader.hxx>
#include <mmx/hash_t.hpp>
#include <mmx/pubkey_t.hpp>

namespace mmx {

struct PlotParams {
    // Core identity
    int ksize = 29;
    int clevel = 0;          // compression level (0-15), 0 = no compression
    bool ssd_mode = false;    // SSD plot (no metadata table) vs HDD (with metadata)
    bool is_nft = false;      // NFT/pool plot vs OG (original)

    // Keys and identity
    hash_t seed;              // random 32-byte seed
    hash_t plot_id;           // SHA256("MMX/PLOTID/OG" || ksize || seed || farmer_key [|| contract])
    pubkey_t farmer_key;      // 33-byte farmer public key
    std::optional<addr_t> contract;  // PlotNFT contract address (for NFT plots)

    // File paths
    std::string plot_name;    // e.g. "plot-mmx-hdd-k29-c0-..."
    std::string tmp_dir;      // temp directory for plot file
    std::string tmp_dir2 = "@RAM";  // partial RAM / disk mode temp
    std::string tmp_dir3 = "@RAM";  // disk mode temp
    std::vector<std::string> final_dirs;  // final destinations (can be multiple, can be @HOST)

    // Derived values (computed by init_derived())
    int xbits = 0;            // bits of X to store = ksize - clevel
    int entry_bits_x = 0;     // line point bits = 2*xbits - 1 (or 2*ksize-1 if uncompressed)
    bool is_hdd_plot = true;  // = !ssd_mode

    void init_derived() {
        xbits = ksize - clevel;
        entry_bits_x = (xbits >= ksize) ? (2 * ksize - 1) : (2 * xbits - 1);
        is_hdd_plot = !ssd_mode;
    }

    bool is_compressed() const {
        return clevel > 0;
    }
};

} // namespace mmx
