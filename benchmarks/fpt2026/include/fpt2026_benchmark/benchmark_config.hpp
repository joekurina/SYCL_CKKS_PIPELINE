#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace fpt2026 {

inline constexpr std::size_t kPolyModulusDegree = 8192;
inline constexpr std::size_t kSlotCount = 4096;
inline constexpr std::size_t kDataModulusCount = 6;
inline constexpr double kScale = 33554432.0; // 2^25
inline constexpr double kCorrectnessThreshold = 0.1;
inline constexpr const char kVectorGeneratorVersion[] = "fpt2026-vector-v1";
inline constexpr const char kBoundaryId[] = "application_e2e_v1";

inline constexpr std::array<std::uint32_t, kDataModulusCount> kDataModuli{
    1053818881u,
    1054015489u,
    1054212097u,
    1055260673u,
    1056178177u,
    1056440321u,
};

inline constexpr std::size_t accelerator_events_per_frame(
    bool save_ntt_s,
    bool save_ntt_pte) noexcept
{
    return 27u + (save_ntt_s ? 12u : 0u) + (save_ntt_pte ? 12u : 0u);
}

static_assert(kPolyModulusDegree == 2u * kSlotCount);
static_assert(kDataModulusCount == kDataModuli.size());

} // namespace fpt2026
