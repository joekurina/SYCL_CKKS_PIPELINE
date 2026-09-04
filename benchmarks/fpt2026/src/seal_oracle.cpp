#include "fpt2026_benchmark/seal_oracle.hpp"

#include "fpt2026_benchmark/provenance.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "seal/seal.h"
#include "seal/util/iterator.h"
#include "seal/util/ntt.h"

namespace fpt2026 {
namespace {

std::vector<std::uint8_t> parms_id_bytes(const seal::parms_id_type& id)
{
    static_assert(sizeof(seal::parms_id_type::value_type) == sizeof(std::uint64_t),
                  "unexpected Microsoft SEAL parms_id word width");
    std::vector<std::uint8_t> bytes;
    bytes.reserve(4 * sizeof(std::uint64_t));
    for (const std::uint64_t word : id) {
        for (unsigned shift = 0; shift < 64; shift += 8) {
            bytes.push_back(static_cast<std::uint8_t>(word >> shift));
        }
    }
    return bytes;
}

template <typename Destination>
void copy_parms_id(const seal::parms_id_type& source, Destination& destination)
{
    if (source.size() != destination.size()) {
        throw std::logic_error("unexpected Microsoft SEAL parms_id size");
    }
    std::copy(source.begin(), source.end(), destination.begin());
}

} // namespace

StockSealOracle::StockSealOracle(
    const std::filesystem::path& serialized_secret_key_path)
{
    const auto key_path = require_absolute_existing_path(serialized_secret_key_path, true);
    parameters_ = std::make_unique<seal::EncryptionParameters>(seal::scheme_type::ckks);
    parameters_->set_poly_modulus_degree(kPolyModulusDegree);

    std::vector<seal::Modulus> moduli;
    moduli.reserve(kDataModulusCount + 1);
    for (const std::uint32_t modulus : kDataModuli) {
        moduli.emplace_back(static_cast<std::uint64_t>(modulus));
    }
    const auto special = seal::CoeffModulus::Create(kPolyModulusDegree, {38});
    if (special.size() != 1) {
        throw std::runtime_error("Microsoft SEAL did not return one 38-bit special prime");
    }
    special_prime_ = special.front().value();
    moduli.push_back(special.front());
    parameters_->set_coeff_modulus(moduli);

    context_ = std::make_unique<seal::SEALContext>(*parameters_);
    if (!context_->parameters_set()) {
        throw std::runtime_error("Microsoft SEAL rejected the FPT 2026 parameter context");
    }
    const auto first_data = context_->first_context_data();
    const auto key_data = context_->key_context_data();
    if (!first_data || !key_data ||
        first_data->parms().coeff_modulus().size() != kDataModulusCount ||
        key_data->parms().coeff_modulus().size() != kDataModulusCount + 1) {
        throw std::runtime_error("Microsoft SEAL context chain does not have six data primes plus one special prime");
    }

    secret_key_ = std::make_unique<seal::SecretKey>();
    std::ifstream key_stream(key_path, std::ios::binary);
    if (!key_stream) {
        throw std::runtime_error("cannot open serialized benchmark secret key: " + key_path.string());
    }
    secret_key_->load(*context_, key_stream);
    if (!key_stream && !key_stream.eof()) {
        throw std::runtime_error("failed while loading serialized benchmark secret key");
    }
    if (secret_key_->parms_id() != context_->key_parms_id() ||
        !secret_key_->data().is_ntt_form()) {
        throw std::runtime_error("serialized benchmark key has the wrong context or representation");
    }

    encoder_ = std::make_unique<seal::CKKSEncoder>(*context_);
    decryptor_ = std::make_unique<seal::Decryptor>(*context_, *secret_key_);
    if (encoder_->slot_count() != kSlotCount) {
        throw std::runtime_error("Microsoft SEAL CKKS encoder does not expose 4096 slots");
    }
}

StockSealOracle::~StockSealOracle() = default;
StockSealOracle::StockSealOracle(StockSealOracle&&) noexcept = default;
StockSealOracle& StockSealOracle::operator=(StockSealOracle&&) noexcept = default;

seal::Ciphertext StockSealOracle::assemble_ciphertext(
    const std::array<std::vector<std::uint32_t>, kDataModulusCount>& c0,
    const std::array<std::vector<std::uint32_t>, kDataModulusCount>& c1) const
{
    for (std::size_t p = 0; p < kDataModulusCount; ++p) {
        if (c0[p].size() != kPolyModulusDegree ||
            c1[p].size() != kPolyModulusDegree) {
            throw std::invalid_argument("each ciphertext residue limb must contain exactly 8192 coefficients");
        }
        for (std::size_t i = 0; i < kPolyModulusDegree; ++i) {
            if (c0[p][i] >= kDataModuli[p] || c1[p][i] >= kDataModuli[p]) {
                throw std::invalid_argument("ciphertext contains a noncanonical residue");
            }
        }
    }

    seal::Ciphertext ciphertext;
    ciphertext.resize(*context_, context_->first_parms_id(), 2);
    if (ciphertext.size() != 2 ||
        ciphertext.poly_modulus_degree() != kPolyModulusDegree ||
        ciphertext.coeff_modulus_size() != kDataModulusCount ||
        ciphertext.parms_id() != context_->first_parms_id()) {
        throw std::runtime_error("Microsoft SEAL created an unexpected ciphertext layout");
    }

    // SEAL stores uint64_t coefficients in polynomial-major, then
    // modulus-major order. Assign each uint32_t residue explicitly; a byte copy
    // across these widths is intentionally forbidden.
    for (std::size_t p = 0; p < kDataModulusCount; ++p) {
        for (std::size_t i = 0; i < kPolyModulusDegree; ++i) {
            ciphertext.data(0)[p * kPolyModulusDegree + i] =
                static_cast<std::uint64_t>(c0[p][i]);
            ciphertext.data(1)[p * kPolyModulusDegree + i] =
                static_cast<std::uint64_t>(c1[p][i]);
        }
    }
    ciphertext.is_ntt_form() = true;
    ciphertext.scale() = kScale;

    if (ciphertext.size() != 2 ||
        ciphertext.coeff_modulus_size() != kDataModulusCount ||
        ciphertext.parms_id() != context_->first_parms_id() ||
        !ciphertext.is_ntt_form() || ciphertext.scale() != kScale) {
        throw std::runtime_error("assembled ciphertext metadata contract failed");
    }
    return ciphertext;
}

std::vector<std::complex<double>> StockSealOracle::decrypt_decode(
    const seal::Ciphertext& ciphertext) const
{
    if (ciphertext.size() != 2 ||
        ciphertext.coeff_modulus_size() != kDataModulusCount ||
        ciphertext.parms_id() != context_->first_parms_id() ||
        !ciphertext.is_ntt_form() || ciphertext.scale() != kScale) {
        throw std::invalid_argument("ciphertext metadata does not match the FPT 2026 oracle contract");
    }
    seal::Plaintext plaintext;
    decryptor_->decrypt(ciphertext, plaintext);
    std::vector<std::complex<double>> decoded;
    encoder_->decode(plaintext, decoded);
    if (decoded.size() != kSlotCount) {
        throw std::runtime_error("Microsoft SEAL decoded an unexpected slot count");
    }
    return decoded;
}

seal::Ciphertext StockSealOracle::encrypt_reference(
    const std::vector<std::complex<double>>& slots) const
{
    if (slots.size() != kSlotCount) {
        throw std::invalid_argument("stock-SEAL reference input must contain exactly 4096 slots");
    }
    seal::Plaintext plaintext;
    encoder_->encode(slots, kScale, plaintext);
    seal::Encryptor encryptor(*context_, *secret_key_);
    seal::Ciphertext ciphertext;
    encryptor.encrypt_symmetric(plaintext, ciphertext);
    return ciphertext;
}

bool StockSealOracle::verify_compact_secret_key(
    const std::filesystem::path& compact_secret_key_path) const
{
    const auto path = require_absolute_existing_path(compact_secret_key_path, true);
    constexpr std::size_t packed_size = kPolyModulusDegree / 4;
    std::array<std::uint8_t, packed_size> packed{};
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(packed.data()),
               static_cast<std::streamsize>(packed.size()));
    if (input.gcount() != static_cast<std::streamsize>(packed.size()) ||
        input.peek() != std::char_traits<char>::eof()) {
        return false;
    }

    seal::Plaintext coefficients = secret_key_->data();
    const auto key_data = context_->key_context_data();
    if (!key_data || !coefficients.is_ntt_form()) {
        return false;
    }
    const auto& moduli = key_data->parms().coeff_modulus();
    seal::util::RNSIter coefficients_iter(coefficients.data(), kPolyModulusDegree);
    seal::util::inverse_ntt_negacyclic_harvey(
        coefficients_iter, moduli.size(), key_data->small_ntt_tables());

    for (std::size_t i = 0; i < kPolyModulusDegree; ++i) {
        const std::uint8_t symbol = static_cast<std::uint8_t>(
            (packed[i / 4] >> (6u - 2u * static_cast<unsigned>(i % 4))) & 0x3u);
        if (symbol == 3u) {
            return false;
        }
        for (std::size_t p = 0; p < moduli.size(); ++p) {
            const std::uint64_t expected = symbol == 0u
                ? moduli[p].value() - 1u
                : static_cast<std::uint64_t>(symbol - 1u);
            if (coefficients[p * kPolyModulusDegree + i] != expected) {
                return false;
            }
        }
    }
    return true;
}

ContextProvenance StockSealOracle::context_provenance(
    std::size_t pipeline_input_block_size,
    const std::string& verification_attempt_id,
    bool key_pair_verified) const
{
    if (pipeline_input_block_size == 0) {
        throw std::invalid_argument("pipeline_input_block_size must be measured or explicitly supplied");
    }
    if (verification_attempt_id.empty()) {
        throw std::invalid_argument("key-pair verification attempt ID is empty");
    }
    if (!key_pair_verified) {
        throw std::runtime_error("key-pair verification did not pass");
    }

    ContextProvenance result;
    result.special_prime = special_prime_;
    const auto key_data = context_->key_context_data();
    const auto first_data = context_->first_context_data();
    if (!key_data || !first_data) {
        throw std::runtime_error("Microsoft SEAL context data is unavailable");
    }
    const auto& key_moduli = key_data->parms().coeff_modulus();
    if (key_moduli.size() != result.key_context_moduli.size()) {
        throw std::runtime_error("unexpected key-context modulus count");
    }
    for (std::size_t i = 0; i < key_moduli.size(); ++i) {
        result.key_context_moduli[i] = key_moduli[i].value();
    }
    copy_parms_id(context_->key_parms_id(), result.key_parms_id_words);
    copy_parms_id(context_->first_parms_id(), result.first_parms_id_words);
    result.key_parms_id_bytes_sha256 = sha256_bytes(
        parms_id_bytes(context_->key_parms_id()));
    result.first_parms_id_bytes_sha256 = sha256_bytes(
        parms_id_bytes(context_->first_parms_id()));
    result.pipeline_input_block_size = pipeline_input_block_size;
    result.uint32_size = sizeof(std::uint32_t);
    result.key_pair_verification.attempt_id = verification_attempt_id;
    result.key_pair_verification.status = "pass";
    return result;
}

const seal::SEALContext& StockSealOracle::context() const noexcept
{
    return *context_;
}

} // namespace fpt2026
