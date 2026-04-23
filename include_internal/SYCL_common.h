#pragma once

#include <cstdint>
#include <cstddef>

namespace sycl_ckks {

constexpr size_t POLY_N = 4096;
constexpr size_t POLY_LOGN = 12;
constexpr size_t LANES = 4;
constexpr size_t NUM_BLOCKS = POLY_N / LANES;
constexpr size_t PIPE_DEPTH_BUFFERED = NUM_BLOCKS;   // Pipes that must buffer a full polynomial
                                                      // (producer far ahead of consumer due to IFFT batch latency)
constexpr size_t PIPE_DEPTH_STREAMING = 64;           // Pipes where producer/consumer run at II=1 in lock-step
                                                      // (compiler may increase beyond this for stall-freedom)
constexpr int MAX_PIPELINES = 3;
constexpr size_t NUM_MODULI = 3;

// ============================================================================
// Barrett const_ratio lookup table for known moduli
// const_ratio = floor(2^64 / q), stored as {low_word, high_word}
// ============================================================================
struct BarrettConstants {
    uint32_t mod_value;
    uint32_t cr_lo;
    uint32_t cr_hi;
};

// Known moduli for n=4096 (27-bit) and n=8192/16384 (30-bit)
constexpr BarrettConstants BARRETT_TABLE[] = {
    // 27-bit primes (n=4096 with SE_DEFAULT_4K_27BIT)
    { 134012929u,  0x0c84dfe5u, 0x00000020u },
    { 134111233u,  0x06814e43u, 0x00000020u },
    { 134176769u,  0x02802e03u, 0x00000020u },
    // 30-bit primes (n=4096 default, n=8192, n=16384)
    // Indices 0-2: Used for n=4096
    { 1053818881u, 0x135bf4bau, 0x00000004u },
    { 1054015489u, 0x132a2218u, 0x00000004u },
    { 1054212097u, 0x12f85437u, 0x00000004u },
    // Indices 3-5: Additional for n=8192 (6 moduli total)
    { 1055260673u, 0x11ef051eu, 0x00000004u },
    { 1056178177u, 0x11074e88u, 0x00000004u },
    { 1056440321u, 0x10c52d4au, 0x00000004u },
    // Indices 6-12: Additional for n=16384 (13 moduli total)
    { 1058209793u, 0x0f07a84au, 0x00000004u },
    { 1060175873u, 0x0d1a6142u, 0x00000004u },
    { 1060700161u, 0x0c9725e9u, 0x00000004u },
    { 1060765697u, 0x0c86c0d4u, 0x00000004u },
    { 1061093377u, 0x0c34cf30u, 0x00000004u },
    { 1062469633u, 0x0add3267u, 0x00000004u },
    { 1062535169u, 0x0accdb49u, 0x00000004u },
};
constexpr size_t BARRETT_TABLE_SIZE = sizeof(BARRETT_TABLE) / sizeof(BARRETT_TABLE[0]);

// Lookup const_ratio for a known modulus (returns false if not found)
inline bool get_barrett_constants(uint32_t mod_value, uint32_t& cr_lo, uint32_t& cr_hi)
{
    // Use switch for better FPGA optimization (enables constant propagation)
    switch (mod_value) {
        case 134012929u:  cr_lo = 0x0c84dfe5u; cr_hi = 0x00000020u; return true;
        case 134111233u:  cr_lo = 0x06814e43u; cr_hi = 0x00000020u; return true;
        case 134176769u:  cr_lo = 0x02802e03u; cr_hi = 0x00000020u; return true;
        case 1053818881u: cr_lo = 0x135bf4bau; cr_hi = 0x00000004u; return true;
        case 1054015489u: cr_lo = 0x132a2218u; cr_hi = 0x00000004u; return true;
        case 1054212097u: cr_lo = 0x12f85437u; cr_hi = 0x00000004u; return true;
        case 1055260673u: cr_lo = 0x11ef051eu; cr_hi = 0x00000004u; return true;
        case 1056178177u: cr_lo = 0x11074e88u; cr_hi = 0x00000004u; return true;
        case 1056440321u: cr_lo = 0x10c52d4au; cr_hi = 0x00000004u; return true;
        case 1058209793u: cr_lo = 0x0f07a84au; cr_hi = 0x00000004u; return true;
        case 1060175873u: cr_lo = 0x0d1a6142u; cr_hi = 0x00000004u; return true;
        case 1060700161u: cr_lo = 0x0c9725e9u; cr_hi = 0x00000004u; return true;
        case 1060765697u: cr_lo = 0x0c86c0d4u; cr_hi = 0x00000004u; return true;
        case 1061093377u: cr_lo = 0x0c34cf30u; cr_hi = 0x00000004u; return true;
        case 1062469633u: cr_lo = 0x0add3267u; cr_hi = 0x00000004u; return true;
        case 1062535169u: cr_lo = 0x0accdb49u; cr_hi = 0x00000004u; return true;
        default: return false;
    }
}

// ============================================================================
// Specialized Barrett reduction using hardcoded const_ratio
// ============================================================================

inline uint32_t barrett_reduce_64_core(
    int64_t val,
    uint32_t mod_value,
    uint32_t cr0,
    uint32_t cr1,
    bool negate_result = false)
{
    uint64_t coeff_abs = (val < 0) ? static_cast<uint64_t>(-val) : static_cast<uint64_t>(val);
    uint32_t sign_mask = static_cast<uint32_t>(val < 0);

    uint32_t coeff_lo = static_cast<uint32_t>(coeff_abs);
    uint32_t coeff_hi = static_cast<uint32_t>(coeff_abs >> 32);

    uint64_t tmp0 = static_cast<uint64_t>(coeff_lo) * cr0;
    uint64_t tmp1 = static_cast<uint64_t>(coeff_lo) * cr1;
    uint64_t tmp2 = static_cast<uint64_t>(coeff_hi) * cr0;

    uint32_t right_hw = static_cast<uint32_t>(tmp0 >> 32);
    uint32_t mid_lo = right_hw + static_cast<uint32_t>(tmp1);
    uint32_t mid_hi = static_cast<uint32_t>(tmp1 >> 32) + (mid_lo < right_hw);
    uint32_t mid2_lo = mid_lo + static_cast<uint32_t>(tmp2);
    uint32_t mid2_hi = static_cast<uint32_t>(tmp2 >> 32) + (mid2_lo < mid_lo);

    uint32_t tmp = coeff_hi * cr1 + mid_hi + mid2_hi;
    tmp = coeff_lo - tmp * mod_value;

    if (tmp >= mod_value) tmp -= mod_value;

    uint32_t result = ((mod_value - tmp) & (-sign_mask)) + (tmp & (sign_mask - 1));

    if (negate_result) {
        uint32_t mask = static_cast<uint32_t>(-(result != 0));
        result = (mod_value - result) & mask;
    }

    return result;
}

inline uint32_t barrett_reduce_u64_core(
    uint64_t product,
    uint32_t mod_value,
    uint32_t cr0,
    uint32_t cr1)
{
    uint32_t prod_lo = static_cast<uint32_t>(product);
    uint32_t prod_hi = static_cast<uint32_t>(product >> 32);

    uint64_t tmp0 = static_cast<uint64_t>(prod_lo) * cr0;
    uint64_t tmp1 = static_cast<uint64_t>(prod_lo) * cr1;
    uint64_t tmp2 = static_cast<uint64_t>(prod_hi) * cr0;

    uint32_t right_hw = static_cast<uint32_t>(tmp0 >> 32);
    uint32_t mid_lo = right_hw + static_cast<uint32_t>(tmp1);
    uint32_t mid_hi = static_cast<uint32_t>(tmp1 >> 32) + (mid_lo < right_hw);
    uint32_t mid2_lo = mid_lo + static_cast<uint32_t>(tmp2);
    uint32_t mid2_hi = static_cast<uint32_t>(tmp2 >> 32) + (mid2_lo < mid_lo);

    uint32_t tmp = prod_hi * cr1 + mid_hi + mid2_hi;
    tmp = prod_lo - tmp * mod_value;

    if (tmp >= mod_value) tmp -= mod_value;
    return tmp;
}

inline uint32_t barrett_reduce_64(
    int64_t val,
    uint32_t mod_value,
    const uint32_t* const_ratio,
    bool negate_result = false)
{
    return barrett_reduce_64_core(val, mod_value, const_ratio[0], const_ratio[1], negate_result);
}

inline uint32_t barrett_reduce_u64(
    uint64_t product,
    uint32_t mod_value,
    const uint32_t* const_ratio)
{
    return barrett_reduce_u64_core(product, mod_value, const_ratio[0], const_ratio[1]);
}

inline uint32_t mod_add(uint32_t a, uint32_t b, uint32_t mod_value)
{
    uint32_t sum = a + b;
    int32_t is_ge_q = static_cast<int32_t>(sum >= mod_value);
    uint32_t mask = static_cast<uint32_t>(-is_ge_q);
    return sum - (mod_value & mask);
}

inline uint32_t mod_neg(uint32_t a, uint32_t mod_value)
{
    int32_t non_zero = static_cast<int32_t>(a != 0);
    uint32_t mask = static_cast<uint32_t>(-non_zero);
    return (mod_value - a) & mask;
}

template <typename T, typename Struct4>
inline T lane_get(const Struct4& block, size_t lane)
{
    switch (lane) {
        case 0: return static_cast<T>(block.element0);
        case 1: return static_cast<T>(block.element1);
        case 2: return static_cast<T>(block.element2);
        case 3: return static_cast<T>(block.element3);
        default: return T{};
    }
}

template <typename T, typename Struct4>
inline void lane_set(Struct4& block, size_t lane, T value)
{
    switch (lane) {
        case 0: block.element0 = value; break;
        case 1: block.element1 = value; break;
        case 2: block.element2 = value; break;
        case 3: block.element3 = value; break;
    }
}

template <typename InStruct, typename OutStruct, typename Func>
inline void lane_transform(const InStruct& in, OutStruct& out, Func&& func)
{
    out.element0 = func(in.element0);
    out.element1 = func(in.element1);
    out.element2 = func(in.element2);
    out.element3 = func(in.element3);
}

template <typename InStruct1, typename InStruct2, typename OutStruct, typename Func>
inline void lane_transform2(const InStruct1& in1, const InStruct2& in2, 
                            OutStruct& out, Func&& func)
{
    out.element0 = func(in1.element0, in2.element0);
    out.element1 = func(in1.element1, in2.element1);
    out.element2 = func(in1.element2, in2.element2);
    out.element3 = func(in1.element3, in2.element3);
}

// Modulus selector for RTL NTT core (maps modulus value to hardware selector index)
inline uint8_t get_modulus_selector(uint32_t mod_value)
{
    switch (mod_value) {
        case 134012929u:  return 0;
        case 134111233u:  return 1;
        case 134176769u:  return 2;
        case 1053818881u: return 3;
        case 1054015489u: return 4;
        case 1054212097u: return 5;
        case 1055260673u: return 6;
        case 1056178177u: return 7;
        case 1056440321u: return 8;
        case 1058209793u: return 9;
        case 1060175873u: return 10;
        case 1060700161u: return 11;
        case 1060765697u: return 12;
        case 1061093377u: return 13;
        case 1062469633u: return 14;
        case 1062535169u: return 15;
        default:          return 0;
    }
}

// NTT primitive root lookup (cryptographic constants for polynomial ring)
inline uint32_t get_ntt_root(size_t n, uint32_t mod_value)
{
    if (n == 4096) {
        switch (mod_value) {
            case 134012929u:  return 7470;
            case 134111233u:  return 3856;
            case 134176769u:  return 24149;
            case 1053818881u: return 503422;
            case 1054015489u: return 16768;
            case 1054212097u: return 7305;
            default:          return 1;
        }
    }
    else if (n == 8192) {
        switch (mod_value) {
            case 1053818881u: return 374229;
            case 1054015489u: return 123363;
            case 1054212097u: return 79941;
            case 1055260673u: return 38869;
            case 1056178177u: return 162146;
            case 1056440321u: return 81884;
            default:          return 1;
        }
    }
    else if (n == 16384) {
        switch (mod_value) {
            case 1053818881u: return 13040;
            case 1054015489u: return 507;
            case 1054212097u: return 1595;
            case 1055260673u: return 68507;
            case 1056178177u: return 3073;
            case 1056440321u: return 6854;
            case 1058209793u: return 44467;
            case 1060175873u: return 16117;
            case 1060700161u: return 27607;
            case 1060765697u: return 222391;
            case 1061093377u: return 105471;
            case 1062469633u: return 310222;
            case 1062535169u: return 2005;
            default:          return 1;
        }
    }
    return 1;
}

}
