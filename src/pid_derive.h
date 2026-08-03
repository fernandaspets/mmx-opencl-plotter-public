#pragma once

#include "plot_params.h"
#include <mmx/hash_t.hpp>
#include <mmx/pubkey_t.hpp>
#include <vnx/vnx.h>
#include <array>
#include <cstring>
#include <stdexcept>

namespace mmx {

// Generate a random 32-byte seed using secure random
inline hash_t generate_seed() {
    hash_t seed;
    vnx::secure_random_bytes(seed.data(), seed.size());
    return seed;
}

// Derive plot_id from seed, ksize, farmer_key, and optional contract.
// Matches reference: SHA256("MMX/PLOTID/OG" || ksize_byte || seed || farmer_key [|| contract])
// For NFT plots: SHA256("MMX/PLOTID/NFT" || ksize_byte || seed || farmer_key || contract)
inline hash_t derive_plot_id(
    const hash_t& seed,
    const int ksize,
    const pubkey_t& farmer_key,
    const bool is_nft,
    const std::optional<addr_t>& contract = std::nullopt)
{
    if(is_nft && !contract) {
        throw std::logic_error("NFT plot requires contract address");
    }

    uint8_t buf[1024] = {};
    uint32_t offset = 0;

    // Tag
    const std::string tag = is_nft ? "MMX/PLOTID/NFT" : "MMX/PLOTID/OG";
    std::memcpy(buf + offset, tag.data(), tag.size());
    offset += tag.size();

    // K-size as single byte
    const uint8_t k = (uint8_t)ksize;
    std::memcpy(buf + offset, &k, 1);
    offset += 1;

    // Seed (32 bytes)
    std::memcpy(buf + offset, seed.data(), seed.size());
    offset += seed.size();

    // Farmer key (33 bytes)
    std::memcpy(buf + offset, farmer_key.data(), farmer_key.size());
    offset += farmer_key.size();

    // Contract (32 bytes, NFT only)
    if(is_nft && contract) {
        std::memcpy(buf + offset, contract->data(), contract->size());
        offset += contract->size();
    }

    return hash_t(buf, offset);
}

// Generate seed + derive plot_id, filling in a PlotParams
inline void generate_plot_identity(PlotParams& params) {
    params.seed = generate_seed();
    params.plot_id = derive_plot_id(
        params.seed, params.ksize, params.farmer_key, params.is_nft, params.contract);
}

} // namespace mmx
