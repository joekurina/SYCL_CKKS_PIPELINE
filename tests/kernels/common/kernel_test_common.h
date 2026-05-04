#pragma once

#include "SYCL_common.h"
#include "SYCL_data_types.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace sycl_ckks::test {

inline u32x4 make_u32x4(uint32_t base)
{
    u32x4 value{};
    value.element0 = base + 0;
    value.element1 = base + 1;
    value.element2 = base + 2;
    value.element3 = base + 3;
    return value;
}

inline i8x4 make_i8x4(int base)
{
    i8x4 value{};
    value.element0 = static_cast<int8_t>(base + 0);
    value.element1 = static_cast<int8_t>(base + 1);
    value.element2 = static_cast<int8_t>(base + 2);
    value.element3 = static_cast<int8_t>(base + 3);
    return value;
}

inline encoding_block make_encoding_block(double base)
{
    encoding_block value{};
    value.element0 = complex_double(base + 0.0, -(base + 0.0));
    value.element1 = complex_double(base + 1.0, -(base + 1.0));
    value.element2 = complex_double(base + 2.0, -(base + 2.0));
    value.element3 = complex_double(base + 3.0, -(base + 3.0));
    return value;
}

inline bool equal_u32x4(const u32x4& a, const u32x4& b)
{
    return a.element0 == b.element0 && a.element1 == b.element1 &&
           a.element2 == b.element2 && a.element3 == b.element3;
}

inline bool equal_i8x4(const i8x4& a, const i8x4& b)
{
    return a.element0 == b.element0 && a.element1 == b.element1 &&
           a.element2 == b.element2 && a.element3 == b.element3;
}

inline bool equal_encoding_block(const encoding_block& a, const encoding_block& b, double tolerance = 0.0)
{
    auto close = [=](complex_double x, complex_double y) {
        return std::abs(x.real() - y.real()) <= tolerance &&
               std::abs(x.imag() - y.imag()) <= tolerance;
    };
    return close(a.element0, b.element0) && close(a.element1, b.element1) &&
           close(a.element2, b.element2) && close(a.element3, b.element3);
}

inline uint32_t lane_at(const u32x4& value, size_t lane)
{
    switch (lane) {
        case 0: return value.element0;
        case 1: return value.element1;
        case 2: return value.element2;
        case 3: return value.element3;
        default: return 0;
    }
}

inline int8_t lane_at(const i8x4& value, size_t lane)
{
    switch (lane) {
        case 0: return value.element0;
        case 1: return value.element1;
        case 2: return value.element2;
        case 3: return value.element3;
        default: return 0;
    }
}

inline complex_double lane_at(const encoding_block& value, size_t lane)
{
    switch (lane) {
        case 0: return value.element0;
        case 1: return value.element1;
        case 2: return value.element2;
        case 3: return value.element3;
        default: return {};
    }
}

inline void set_lane(u32x4& value, size_t lane, uint32_t lane_value)
{
    switch (lane) {
        case 0: value.element0 = lane_value; break;
        case 1: value.element1 = lane_value; break;
        case 2: value.element2 = lane_value; break;
        case 3: value.element3 = lane_value; break;
    }
}

inline bool expect_u32x4_vector(const char* label,
                               const std::vector<u32x4>& expected,
                               const std::vector<u32x4>& actual)
{
    if (expected.size() != actual.size()) {
        std::cerr << label << " size mismatch expected=" << expected.size()
                  << " actual=" << actual.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!equal_u32x4(expected[i], actual[i])) {
            std::cerr << label << " mismatch block=" << i
                      << " expected={" << expected[i].element0 << "," << expected[i].element1 << ","
                      << expected[i].element2 << "," << expected[i].element3 << "} actual={"
                      << actual[i].element0 << "," << actual[i].element1 << ","
                      << actual[i].element2 << "," << actual[i].element3 << "}\n";
            return false;
        }
    }
    return true;
}

inline bool expect_i8x4_vector(const char* label,
                              const std::vector<i8x4>& expected,
                              const std::vector<i8x4>& actual)
{
    if (expected.size() != actual.size()) {
        std::cerr << label << " size mismatch expected=" << expected.size()
                  << " actual=" << actual.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!equal_i8x4(expected[i], actual[i])) {
            std::cerr << label << " mismatch block=" << i << "\n";
            return false;
        }
    }
    return true;
}

inline bool expect_encoding_vector(const char* label,
                                  const std::vector<encoding_block>& expected,
                                  const std::vector<encoding_block>& actual,
                                  double tolerance = 0.0)
{
    if (expected.size() != actual.size()) {
        std::cerr << label << " size mismatch expected=" << expected.size()
                  << " actual=" << actual.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!equal_encoding_block(expected[i], actual[i], tolerance)) {
            std::cerr << label << " mismatch block=" << i << "\n";
            return false;
        }
    }
    return true;
}

inline std::vector<u32x4> make_u32x4_sequence(uint32_t base)
{
    std::vector<u32x4> values(NUM_BLOCKS);
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        values[i] = make_u32x4(base + static_cast<uint32_t>(i * LANES));
    }
    return values;
}

inline uint32_t test_modulus(size_t p)
{
    return BARRETT_TABLE[p].mod_value;
}

inline void test_const_ratio(size_t p, uint32_t out[2])
{
    out[0] = BARRETT_TABLE[p].cr_lo;
    out[1] = BARRETT_TABLE[p].cr_hi;
}

}  // namespace sycl_ckks::test
