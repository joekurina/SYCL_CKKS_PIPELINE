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

// The harnesses keep the input data deliberately simple. Each test first
// creates a host-side vector where element i is derived directly from i, then
// passes that vector to a feeder kernel through a SYCL buffer. The helpers
// below only adapt that scalar index to the aggregate lane types used by the
// production pipeline.
inline uint32_t test_scalar(size_t i, uint32_t offset = 0)
{
    return static_cast<uint32_t>(i) + offset;
}

inline u32x4 make_test_u32x4(size_t i, uint32_t offset = 0)
{
    u32x4 out;
    out.element0 = test_scalar(i, offset + 0u);
    out.element1 = test_scalar(i, offset + 1u);
    out.element2 = test_scalar(i, offset + 2u);
    out.element3 = test_scalar(i, offset + 3u);
    return out;
}

inline i8x4 make_test_i8x4(size_t i)
{
    i8x4 out;
    out.element0 = static_cast<int8_t>(static_cast<int>(i) + 0);
    out.element1 = static_cast<int8_t>(static_cast<int>(i) + 1);
    out.element2 = static_cast<int8_t>(static_cast<int>(i) + 2);
    out.element3 = static_cast<int8_t>(static_cast<int>(i) + 3);
    return out;
}

inline encoding_block make_test_encoding(size_t i)
{
    const double value = static_cast<double>(i);
    encoding_block out;
    out.element0 = complex_double(value + 0.0, 0.0);
    out.element1 = complex_double(value + 1.0, 0.0);
    out.element2 = complex_double(value + 2.0, 0.0);
    out.element3 = complex_double(value + 3.0, 0.0);
    return out;
}

inline PipelineInputBlock make_test_input_block(size_t i)
{
    PipelineInputBlock out{};
    out.encoding = make_test_encoding(i);
    out.error = make_test_i8x4(i);
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        const uint32_t per_modulus_offset = static_cast<uint32_t>(100u * (p + 1u));
        out.secret_key[p] = make_test_u32x4(i, per_modulus_offset);
        out.c1[p] = make_test_u32x4(i, per_modulus_offset + 50u);
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
