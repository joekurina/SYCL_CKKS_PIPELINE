#include "fpt2026_benchmark/seal_embedded_runner.hpp"

#include "fpt2026_benchmark/seal_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fpt2026 {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end)
{
    const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    if (value < 0) {
        throw std::runtime_error("steady_clock produced a negative interval");
    }
    return static_cast<std::uint64_t>(value);
}

void check_c_status(int status, const char* error_message, const char* operation)
{
    if (status != 0) {
        throw std::runtime_error(
            std::string(operation) + " failed: " +
            (error_message && error_message[0] ? error_message : "unknown C shim error"));
    }
}

std::vector<std::uint8_t> concatenate_seeds(const std::vector<TrialSeeds>& seeds, bool shareable)
{
    std::vector<std::uint8_t> result;
    result.reserve(seeds.size() * 64);
    for (const auto& seed : seeds) {
        const auto& source = shareable ? seed.shareable : seed.error;
        result.insert(result.end(), source.begin(), source.end());
    }
    return result;
}

std::string combine_seed_digest(
    const std::array<std::uint8_t, 64>& shareable,
    const std::array<std::uint8_t, 64>& error)
{
    std::vector<std::uint8_t> bytes;
    bytes.reserve(128);
    bytes.insert(bytes.end(), shareable.begin(), shareable.end());
    bytes.insert(bytes.end(), error.begin(), error.end());
    return sha256_bytes(bytes);
}

} // namespace

TrialSeeds make_trial_seeds(
    const std::string& run_id,
    const std::string& case_id,
    std::size_t repetition,
    std::size_t frame_index)
{
    TrialSeeds result;
    result.shareable = derive_trial_seed(
        TrialDescriptor{run_id, case_id, repetition, frame_index, "uniform-c1", -1});
    result.error = derive_trial_seed(
        TrialDescriptor{run_id, case_id, repetition, frame_index, "ckks-error", -1});
    result.digest = combine_seed_digest(result.shareable, result.error);
    return result;
}

SealEmbeddedRunner::SealEmbeddedRunner(
    const std::filesystem::path& compact_secret_key_path,
    StockSealOracle& oracle)
    : oracle_(&oracle), index_map_(kPolyModulusDegree)
{
    const auto key_path = require_absolute_existing_path(compact_secret_key_path, true);
    const std::string key_string = key_path.string();
    FPT2026SealEmbeddedConfig config{};
    config.abi_version = FPT2026_SEAL_EMBEDDED_C_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.degree = kPolyModulusDegree;
    config.num_moduli = kDataModulusCount;
    config.scale = kScale;
    config.compact_key_path = key_string.c_str();
    char error[512]{};
    check_c_status(
        fpt2026_seal_embedded_create(&config, &state_, error, sizeof(error)),
        error, "fpt2026_seal_embedded_create");

    try {
        check_c_status(
            fpt2026_seal_embedded_copy_parameters(
                state_, moduli_.data(), moduli_.size(), const_ratios_.data(),
                const_ratios_.size(), error, sizeof(error)),
            error, "fpt2026_seal_embedded_copy_parameters");
        check_c_status(
            fpt2026_seal_embedded_copy_index_map(
                state_, index_map_.data(), index_map_.size(), error, sizeof(error)),
            error, "fpt2026_seal_embedded_copy_index_map");
        if (moduli_ != kDataModuli) {
            throw std::runtime_error("SEAL-Embedded returned the wrong data moduli");
        }
    } catch (...) {
        fpt2026_seal_embedded_destroy(state_);
        state_ = nullptr;
        throw;
    }
}

SealEmbeddedRunner::~SealEmbeddedRunner()
{
    fpt2026_seal_embedded_destroy(state_);
}

BackendBatchResult SealEmbeddedRunner::encrypt(
    const std::vector<BenchmarkVector>& frames,
    const std::vector<TrialSeeds>& seeds)
{
    if (frames.empty() || frames.size() != seeds.size()) {
        throw std::invalid_argument("CPU frame and seed counts must be equal and nonzero");
    }
    for (const auto& frame : frames) {
        if (is_complex_case(frame.case_id)) {
            throw std::invalid_argument("SEAL-Embedded CPU encryption is real-only");
        }
    }

    BackendBatchResult result;
    result.residues.frame_count = frames.size();
    for (std::size_t p = 0; p < kDataModulusCount; ++p) {
        result.residues.c0[p].resize(frames.size() * kPolyModulusDegree);
        result.residues.c1[p].resize(frames.size() * kPolyModulusDegree);
    }
    result.ciphertexts.reserve(frames.size());
    result.trial_seed_digests.reserve(frames.size());

    const auto application_begin = Clock::now();
    const auto preparation_begin = application_begin;
    std::vector<std::vector<float>> values(frames.size());
    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        if (frames[frame].slots.size() != kSlotCount) {
            throw std::invalid_argument("CPU input does not contain exactly 4096 slots");
        }
        values[frame].resize(kSlotCount);
        for (std::size_t i = 0; i < kSlotCount; ++i) {
            if (frames[frame].slots[i].imag() != 0.0) {
                throw std::invalid_argument("SEAL-Embedded CPU input contains an imaginary component");
            }
            values[frame][i] = static_cast<float>(frames[frame].slots[i].real());
        }
        result.trial_seed_digests.push_back(seeds[frame].digest);
    }
    const auto preparation_end = Clock::now();
    result.timing.preparation_wall_ns = elapsed_ns(preparation_begin, preparation_end);

    std::vector<std::uint32_t> frame_c0(kDataModulusCount * kPolyModulusDegree);
    std::vector<std::uint32_t> frame_c1(kDataModulusCount * kPolyModulusDegree);
    char error[512]{};
    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        FPT2026SealEmbeddedRunResult c_result{};
        c_result.abi_version = FPT2026_SEAL_EMBEDDED_C_ABI_VERSION;
        c_result.struct_size = sizeof(c_result);
        error[0] = '\0';
        check_c_status(
            fpt2026_seal_embedded_encrypt(
                state_, values[frame].data(), values[frame].size(),
                seeds[frame].shareable.data(), seeds[frame].error.data(),
                frame_c0.data(), frame_c0.size(), frame_c1.data(), frame_c1.size(),
                &c_result, error, sizeof(error)),
            error, "fpt2026_seal_embedded_encrypt");
        if (c_result.modulus_visit_count != kDataModulusCount) {
            throw std::runtime_error("SEAL-Embedded did not visit exactly six moduli");
        }
        for (std::size_t p = 0; p < kDataModulusCount; ++p) {
            if (c_result.modulus_visits[p] != p) {
                throw std::runtime_error("SEAL-Embedded modulus iterator did not visit 0 through 5 exactly once");
            }
            std::copy_n(frame_c0.data() + p * kPolyModulusDegree,
                        kPolyModulusDegree,
                        result.residues.c0[p].data() + frame * kPolyModulusDegree);
            std::copy_n(frame_c1.data() + p * kPolyModulusDegree,
                        kPolyModulusDegree,
                        result.residues.c1[p].data() + frame * kPolyModulusDegree);
        }
    }

    const auto assembly_begin = Clock::now();
    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        std::array<std::vector<std::uint32_t>, kDataModulusCount> c0;
        std::array<std::vector<std::uint32_t>, kDataModulusCount> c1;
        for (std::size_t p = 0; p < kDataModulusCount; ++p) {
            const auto first0 = result.residues.c0[p].begin() +
                                static_cast<std::ptrdiff_t>(frame * kPolyModulusDegree);
            const auto first1 = result.residues.c1[p].begin() +
                                static_cast<std::ptrdiff_t>(frame * kPolyModulusDegree);
            c0[p].assign(first0, first0 + static_cast<std::ptrdiff_t>(kPolyModulusDegree));
            c1[p].assign(first1, first1 + static_cast<std::ptrdiff_t>(kPolyModulusDegree));
        }
        result.ciphertexts.push_back(oracle_->assemble_ciphertext(c0, c1));
    }
    const auto application_end = Clock::now();
    result.timing.unpack_assembly_wall_ns = elapsed_ns(assembly_begin, application_end);
    result.timing.application_e2e_ns = elapsed_ns(application_begin, application_end);
    const std::uint64_t attributed = result.timing.preparation_wall_ns +
                                     result.timing.unpack_assembly_wall_ns;
    result.timing.unattributed_wall_ns =
        result.timing.application_e2e_ns >= attributed
            ? result.timing.application_e2e_ns - attributed
            : 0;

    // Correctness is deliberately after the application timer.
    result.passed = true;
    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        auto decoded = oracle_->decrypt_decode(result.ciphertexts[frame]);
        auto metrics = evaluate_slots(
            frames[frame].slots, decoded, frames[frame].active_slots,
            kCorrectnessThreshold);
        result.passed = result.passed && metrics.passed;
        result.decoded_slots.push_back(std::move(decoded));
        result.correctness.push_back(std::move(metrics));
    }
    return result;
}

PreparedAcceleratorInputs SealEmbeddedRunner::prepare_accelerator_inputs(
    const std::vector<BenchmarkVector>& frames,
    const std::vector<TrialSeeds>& seeds)
{
    if (frames.empty() || frames.size() != seeds.size()) {
        throw std::invalid_argument("accelerator frame and seed counts must be equal and nonzero");
    }
    PreparedAcceleratorInputs result;
    result.frame_count = frames.size();
    result.encoding_buffers.reserve(frames.size() * kPolyModulusDegree);
    for (const auto& frame : frames) {
        auto embedded = make_conjugate_embedding(frame, index_map_);
        result.encoding_buffers.insert(
            result.encoding_buffers.end(), embedded.begin(), embedded.end());
    }
    result.error_samples.resize(frames.size() * kPolyModulusDegree);
    for (std::size_t p = 0; p < kDataModulusCount; ++p) {
        result.secret_keys[p].resize(frames.size() * kPolyModulusDegree);
        result.uniform_polys[p].resize(frames.size() * kPolyModulusDegree);
    }

    const auto shareable = concatenate_seeds(seeds, true);
    const auto error = concatenate_seeds(seeds, false);
    std::vector<std::uint32_t> secret_flat(
        kDataModulusCount * frames.size() * kPolyModulusDegree);
    std::vector<std::uint32_t> uniform_flat(secret_flat.size());
    char message[512]{};
    check_c_status(
        fpt2026_seal_embedded_prepare_accelerator_inputs(
            state_, frames.size(), shareable.data(), shareable.size(),
            error.data(), error.size(), result.error_samples.data(),
            result.error_samples.size(), secret_flat.data(), secret_flat.size(),
            uniform_flat.data(), uniform_flat.size(), message, sizeof(message)),
        message, "fpt2026_seal_embedded_prepare_accelerator_inputs");

    const std::size_t per_modulus = frames.size() * kPolyModulusDegree;
    for (std::size_t p = 0; p < kDataModulusCount; ++p) {
        std::copy_n(secret_flat.data() + p * per_modulus, per_modulus,
                    result.secret_keys[p].data());
        std::copy_n(uniform_flat.data() + p * per_modulus, per_modulus,
                    result.uniform_polys[p].data());
    }
    return result;
}

const std::vector<std::uint16_t>& SealEmbeddedRunner::index_map() const noexcept
{
    return index_map_;
}

const std::array<std::uint32_t, kDataModulusCount>&
SealEmbeddedRunner::moduli() const noexcept
{
    return moduli_;
}

const std::array<std::uint32_t, 2 * kDataModulusCount>&
SealEmbeddedRunner::const_ratios() const noexcept
{
    return const_ratios_;
}

} // namespace fpt2026
