#include "fpt2026_benchmark/benchmark_vectors.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fpt2026 {
namespace {

constexpr std::size_t kShake256Rate = 136;

constexpr std::array<std::uint64_t, 24> kRoundConstants{
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

constexpr std::array<unsigned, 25> kRotation{
    0, 1, 62, 28, 27,
    36, 44, 6, 55, 20,
    3, 10, 43, 25, 39,
    41, 45, 15, 21, 8,
    18, 2, 61, 56, 14,
};

std::uint64_t rotate_left(std::uint64_t value, unsigned count) noexcept
{
    return count == 0 ? value : (value << count) | (value >> (64u - count));
}

void keccak_f1600(std::array<std::uint64_t, 25>& state) noexcept
{
    for (const std::uint64_t round_constant : kRoundConstants) {
        std::array<std::uint64_t, 5> c{};
        std::array<std::uint64_t, 5> d{};
        std::array<std::uint64_t, 25> b{};

        for (std::size_t x = 0; x < 5; ++x) {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^
                   state[x + 15] ^ state[x + 20];
        }
        for (std::size_t x = 0; x < 5; ++x) {
            d[x] = c[(x + 4) % 5] ^ rotate_left(c[(x + 1) % 5], 1);
        }
        for (std::size_t y = 0; y < 5; ++y) {
            for (std::size_t x = 0; x < 5; ++x) {
                state[x + 5 * y] ^= d[x];
            }
        }

        for (std::size_t y = 0; y < 5; ++y) {
            for (std::size_t x = 0; x < 5; ++x) {
                const std::size_t new_x = y;
                const std::size_t new_y = (2 * x + 3 * y) % 5;
                b[new_x + 5 * new_y] = rotate_left(
                    state[x + 5 * y], kRotation[x + 5 * y]);
            }
        }

        for (std::size_t y = 0; y < 5; ++y) {
            for (std::size_t x = 0; x < 5; ++x) {
                state[x + 5 * y] = b[x + 5 * y] ^
                    ((~b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);
            }
        }
        state[0] ^= round_constant;
    }
}

std::uint8_t state_byte(
    const std::array<std::uint64_t, 25>& state,
    std::size_t offset) noexcept
{
    return static_cast<std::uint8_t>(
        state[offset / 8] >> (8u * static_cast<unsigned>(offset % 8)));
}

void xor_state_byte(
    std::array<std::uint64_t, 25>& state,
    std::size_t offset,
    std::uint8_t value) noexcept
{
    state[offset / 8] ^=
        static_cast<std::uint64_t>(value) << (8u * static_cast<unsigned>(offset % 8));
}

std::uint32_t load_u32_le(const std::uint8_t* bytes) noexcept
{
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8u) |
           (static_cast<std::uint32_t>(bytes[2]) << 16u) |
           (static_cast<std::uint32_t>(bytes[3]) << 24u);
}

double generated_component(
    const std::vector<std::uint8_t>& bytes,
    std::size_t index)
{
    const std::uint32_t word = load_u32_le(bytes.data() + 4 * index);
    std::int64_t centered = static_cast<std::int64_t>(word % 2000001u) - 1000000;
    if (centered == 0) {
        centered = 1;
    }
    return static_cast<double>(centered) / 1000000.0;
}

void append_u64_le(std::vector<std::uint8_t>& output, std::uint64_t value)
{
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void append_double_le(std::vector<std::uint8_t>& output, double value)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "vector artifacts require IEEE-754 binary64 doubles");
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u64_le(output, bits);
}

} // namespace

std::vector<std::uint8_t> shake256(
    std::string_view domain,
    std::size_t output_size)
{
    std::array<std::uint64_t, 25> state{};
    std::size_t offset = 0;
    for (const unsigned char byte : domain) {
        xor_state_byte(state, offset, static_cast<std::uint8_t>(byte));
        ++offset;
        if (offset == kShake256Rate) {
            keccak_f1600(state);
            offset = 0;
        }
    }
    xor_state_byte(state, offset, 0x1fU);
    xor_state_byte(state, kShake256Rate - 1, 0x80U);
    keccak_f1600(state);

    std::vector<std::uint8_t> output(output_size);
    for (std::size_t i = 0; i < output_size; ++i) {
        if (i != 0 && i % kShake256Rate == 0) {
            keccak_f1600(state);
        }
        output[i] = state_byte(state, i % kShake256Rate);
    }
    return output;
}

VectorCaseId parse_vector_case(std::string_view id)
{
    for (const VectorCaseId candidate : all_vector_cases()) {
        if (vector_case_name(candidate) == id) {
            return candidate;
        }
    }
    throw std::invalid_argument("unknown FPT 2026 vector case: " + std::string(id));
}

std::string_view vector_case_name(VectorCaseId id) noexcept
{
    switch (id) {
    case VectorCaseId::real_impulse: return "real_impulse";
    case VectorCaseId::real_short_mixed: return "real_short_mixed";
    case VectorCaseId::real_partial_64: return "real_partial_64";
    case VectorCaseId::real_partial_1024: return "real_partial_1024";
    case VectorCaseId::real_full_4096: return "real_full_4096";
    case VectorCaseId::complex_short_mixed: return "complex_short_mixed";
    case VectorCaseId::complex_partial_64: return "complex_partial_64";
    case VectorCaseId::complex_full_4096: return "complex_full_4096";
    }
    return "unknown";
}

bool is_complex_case(VectorCaseId id) noexcept
{
    return id == VectorCaseId::complex_short_mixed ||
           id == VectorCaseId::complex_partial_64 ||
           id == VectorCaseId::complex_full_4096;
}

std::size_t active_slot_count(VectorCaseId id) noexcept
{
    switch (id) {
    case VectorCaseId::real_impulse: return 1;
    case VectorCaseId::real_short_mixed:
    case VectorCaseId::complex_short_mixed: return 8;
    case VectorCaseId::real_partial_64:
    case VectorCaseId::complex_partial_64: return 64;
    case VectorCaseId::real_partial_1024: return 1024;
    case VectorCaseId::real_full_4096:
    case VectorCaseId::complex_full_4096: return kSlotCount;
    }
    return 0;
}

std::vector<VectorCaseId> all_vector_cases()
{
    return {
        VectorCaseId::real_impulse,
        VectorCaseId::real_short_mixed,
        VectorCaseId::real_partial_64,
        VectorCaseId::real_partial_1024,
        VectorCaseId::real_full_4096,
        VectorCaseId::complex_short_mixed,
        VectorCaseId::complex_partial_64,
        VectorCaseId::complex_full_4096,
    };
}

BenchmarkVector generate_benchmark_vector(VectorCaseId id)
{
    BenchmarkVector result;
    result.case_id = id;
    result.id = std::string(vector_case_name(id));
    result.active_slots = active_slot_count(id);
    result.slots.assign(kSlotCount, {0.0, 0.0});

    if (id == VectorCaseId::real_impulse) {
        result.slots[0] = {1.0, 0.0};
        return result;
    }

    if (id == VectorCaseId::real_short_mixed) {
        constexpr std::array<double, 8> values{
            1.25, -0.5, 0.125, -1.75, 2.5, -0.03125, 0.875, -2.125};
        for (std::size_t i = 0; i < values.size(); ++i) {
            result.slots[i] = {values[i], 0.0};
        }
        return result;
    }

    if (id == VectorCaseId::complex_short_mixed) {
        constexpr std::array<std::complex<double>, 8> values{{
            {1.25, 0.5}, {-0.5, 1.125}, {0.125, -0.75}, {-1.75, -0.25},
            {2.5, 0.03125}, {-0.03125, 2.0}, {0.875, -1.5}, {-2.125, 0.625},
        }};
        for (std::size_t i = 0; i < values.size(); ++i) {
            result.slots[i] = values[i];
        }
        return result;
    }

    const std::string prefix = "FPT2026/vector/v1/" + result.id + "/";
    const auto real_bytes = shake256(prefix + "real", 4 * result.active_slots);
    std::vector<std::uint8_t> imag_bytes;
    if (is_complex_case(id)) {
        imag_bytes = shake256(prefix + "imag", 4 * result.active_slots);
    }
    for (std::size_t i = 0; i < result.active_slots; ++i) {
        const double real = generated_component(real_bytes, i);
        const double imag = imag_bytes.empty() ? 0.0 : generated_component(imag_bytes, i);
        result.slots[i] = {real, imag};
    }
    return result;
}

std::vector<std::complex<double>> make_conjugate_embedding(
    const BenchmarkVector& input,
    const std::vector<std::uint16_t>& index_map)
{
    if (input.slots.size() != kSlotCount) {
        throw std::invalid_argument("benchmark vector must contain exactly 4096 slots");
    }
    if (input.active_slots > kSlotCount) {
        throw std::invalid_argument("benchmark vector active-slot count exceeds 4096");
    }
    if (index_map.size() != kPolyModulusDegree) {
        throw std::invalid_argument("SEAL-Embedded index map must contain exactly 8192 entries");
    }

    std::vector<std::complex<double>> result(kPolyModulusDegree, {0.0, 0.0});
    for (std::size_t i = 0; i < kSlotCount; ++i) {
        const std::size_t direct = index_map[i];
        const std::size_t conjugate = index_map[i + kSlotCount];
        if (direct >= kPolyModulusDegree || conjugate >= kPolyModulusDegree) {
            throw std::invalid_argument("SEAL-Embedded index map contains an out-of-range entry");
        }
        result[direct] = input.slots[i];
        result[conjugate] = std::conj(input.slots[i]);
    }
    return result;
}

std::vector<std::uint8_t> serialize_vector_artifact(const BenchmarkVector& input)
{
    if (input.slots.size() != kSlotCount || input.active_slots > kSlotCount) {
        throw std::invalid_argument("cannot serialize a malformed benchmark vector");
    }
    std::vector<std::uint8_t> output;
    const std::string header = std::string("FPT2026-VECTOR\0", 15) +
                               kVectorGeneratorVersion + "\0" + input.id + "\0";
    output.insert(output.end(), header.begin(), header.end());
    append_u64_le(output, static_cast<std::uint64_t>(input.active_slots));
    append_u64_le(output, static_cast<std::uint64_t>(input.slots.size()));
    for (const auto& value : input.slots) {
        append_double_le(output, value.real());
        append_double_le(output, value.imag());
    }
    return output;
}

} // namespace fpt2026
