// test_offset_sum.cpp — Verify CPU prefix sum approach
#include <iostream>
#include <vector>
#include <random>
int main() {
    const int N = 2048;
    std::mt19937 rng(123);
    std::vector<uint32_t> counts(N);
    for (auto& c : counts) c = rng() % 100;
    
    std::vector<uint32_t> offset(N + 1, 0);
    for (int i = 0; i < N; i++) offset[i + 1] = offset[i] + counts[i];
    
    // Verify
    bool ok = true;
    for (int i = 1; i <= N; i++) {
        if (offset[i] != offset[i-1] + counts[i-1]) { ok = false; break; }
    }
    std::cout << (ok ? "✅ CPU prefix sum OK" : "❌ FAIL") << std::endl;
    std::cout << "Total entries: " << offset[N] << std::endl;
    return 0;
}
