#pragma once

#include "SYCL_common.h"
#include "SYCL_data_types.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>


namespace sycl_ckks::harness {

inline void host_debug(const char* message)
{
    std::cerr << "[FPGA Test harness host] " << message << '\n';
    std::cerr.flush();
}

inline sycl::queue make_queue()
{
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
    return sycl::queue{selector, sycl::property::queue::enable_profiling()};
}

inline void require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << "[FPGA Test harness] " << message << '\n';
        std::exit(1);
    }
}

inline uint32_t pattern_u32(size_t block, size_t lane, uint32_t salt = 0)
{
    return static_cast<uint32_t>((block * LANES + lane + 1u) * 17u + salt);
}

inline int8_t pattern_i8(size_t block, size_t lane)
{
    return static_cast<int8_t>((static_cast<int>(block + lane) % 7) - 3);
}

inline encoding_block pattern_encoding(size_t block)
{
    const double base = static_cast<double>(block * LANES);
    encoding_block out;
    out.element0 = complex_double(base + 0.25, -base - 0.25);
    out.element1 = complex_double(base + 1.25, -base - 1.25);
    out.element2 = complex_double(base + 2.25, -base - 2.25);
    out.element3 = complex_double(base + 3.25, -base - 3.25);
    return out;
}

inline u32x4 pattern_u32x4(size_t block, uint32_t salt = 0)
{
    u32x4 out;
    out.element0 = pattern_u32(block, 0, salt);
    out.element1 = pattern_u32(block, 1, salt);
    out.element2 = pattern_u32(block, 2, salt);
    out.element3 = pattern_u32(block, 3, salt);
    return out;
}

inline i8x4 pattern_i8x4(size_t block)
{
    i8x4 out;
    out.element0 = pattern_i8(block, 0);
    out.element1 = pattern_i8(block, 1);
    out.element2 = pattern_i8(block, 2);
    out.element3 = pattern_i8(block, 3);
    return out;
}

inline PipelineInputBlock pattern_input_block(size_t block)
{
    PipelineInputBlock out{};
    out.encoding = pattern_encoding(block);
    out.error = pattern_i8x4(block);
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        out.secret_key[p] = pattern_u32x4(block, static_cast<uint32_t>(1000u * (p + 1u)));
        out.c1[p] = pattern_u32x4(block, static_cast<uint32_t>(2000u * (p + 1u)));
    }
    return out;
}

inline bool equal_u32x4(const u32x4& a, const u32x4& b)
{
    return a.element0 == b.element0 &&
           a.element1 == b.element1 &&
           a.element2 == b.element2 &&
           a.element3 == b.element3;
}

inline bool equal_i8x4(const i8x4& a, const i8x4& b)
{
    return a.element0 == b.element0 &&
           a.element1 == b.element1 &&
           a.element2 == b.element2 &&
           a.element3 == b.element3;
}

inline bool equal_encoding(const encoding_block& a, const encoding_block& b)
{
    return a.element0 == b.element0 &&
           a.element1 == b.element1 &&
           a.element2 == b.element2 &&
           a.element3 == b.element3;
}

inline uint32_t default_modulus(size_t p)
{
    static constexpr uint32_t mods[NUM_MODULI] = {
        1053818881u,
        1054015489u,
        1054212097u,
    };
    return mods[p];
}

inline void default_const_ratio(size_t p, uint32_t ratio[2])
{
    bool ok = get_barrett_constants(default_modulus(p), ratio[0], ratio[1]);
    require(ok, "default modulus is missing Barrett constants");
}

} // namespace sycl_ckks::harness
