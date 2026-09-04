#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "fpt2026_benchmark/benchmark_config.hpp"

namespace seal {
class Ciphertext;
class CKKSEncoder;
class Decryptor;
class EncryptionParameters;
class SEALContext;
class SecretKey;
} // namespace seal

namespace fpt2026 {

struct KeyPairVerification {
    std::string attempt_id;
    std::string status;
};

struct ContextProvenance {
    std::uint64_t special_prime{};
    std::array<std::uint64_t, 7> key_context_moduli{};
    std::array<std::uint64_t, 4> key_parms_id_words{};
    std::array<std::uint64_t, 4> first_parms_id_words{};
    std::string key_parms_id_bytes_sha256;
    std::string first_parms_id_bytes_sha256;
    std::size_t ciphertext_size{2};
    std::size_t ciphertext_modulus_count{6};
    bool ciphertext_is_ntt_form{true};
    std::size_t pipeline_input_block_size{};
    std::size_t uint32_size{sizeof(std::uint32_t)};
    KeyPairVerification key_pair_verification;
};

class StockSealOracle {
public:
    explicit StockSealOracle(const std::filesystem::path& serialized_secret_key_path);
    ~StockSealOracle();

    StockSealOracle(const StockSealOracle&) = delete;
    StockSealOracle& operator=(const StockSealOracle&) = delete;
    StockSealOracle(StockSealOracle&&) noexcept;
    StockSealOracle& operator=(StockSealOracle&&) noexcept;

    seal::Ciphertext assemble_ciphertext(
        const std::array<std::vector<std::uint32_t>, kDataModulusCount>& c0,
        const std::array<std::vector<std::uint32_t>, kDataModulusCount>& c1) const;

    std::vector<std::complex<double>> decrypt_decode(
        const seal::Ciphertext& ciphertext) const;

    seal::Ciphertext encrypt_reference(
        const std::vector<std::complex<double>>& slots) const;

    // Compares the adapter's packed 2-bit coefficient-domain key against the
    // serialized stock-SEAL key after an independent inverse NTT.
    bool verify_compact_secret_key(
        const std::filesystem::path& compact_secret_key_path) const;

    ContextProvenance context_provenance(
        std::size_t pipeline_input_block_size,
        const std::string& verification_attempt_id,
        bool key_pair_verified) const;

    const seal::SEALContext& context() const noexcept;

private:
    std::unique_ptr<seal::EncryptionParameters> parameters_;
    std::unique_ptr<seal::SEALContext> context_;
    std::unique_ptr<seal::SecretKey> secret_key_;
    std::unique_ptr<seal::CKKSEncoder> encoder_;
    std::unique_ptr<seal::Decryptor> decryptor_;
    std::uint64_t special_prime_{};
};

} // namespace fpt2026
