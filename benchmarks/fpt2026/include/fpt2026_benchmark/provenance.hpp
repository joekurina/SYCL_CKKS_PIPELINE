#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace fpt2026 {

std::string sha256_bytes(const std::uint8_t* data, std::size_t size);
std::string sha256_bytes(const std::vector<std::uint8_t>& data);
std::string sha256_file(const std::filesystem::path& path);
std::string utc_now_iso8601();
std::filesystem::path require_absolute_existing_path(
    const std::filesystem::path& path,
    bool require_regular_file);

struct TrialDescriptor {
    std::string run_id;
    std::string case_id;
    std::size_t repetition{};
    std::size_t frame_index{};
    std::string purpose;
    std::int32_t prime_index{-1};
};

std::string trial_seed_domain(const TrialDescriptor& descriptor);
std::array<std::uint8_t, 64> derive_trial_seed(const TrialDescriptor& descriptor);

struct BuildProvenance {
    std::string cxx_compiler;
    std::string cxx_standard;
    std::string accelerator_commit;
    std::string seal_embedded_commit;
    std::string microsoft_seal_commit;
};

BuildProvenance compiled_build_provenance();

} // namespace fpt2026
