#include "fpt2026_benchmark/provenance.hpp"

#include "fpt2026_benchmark/benchmark_vectors.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace fpt2026 {
namespace {

constexpr std::array<std::uint32_t, 64> kSha256Constants{
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

std::uint32_t rotr(std::uint32_t value, unsigned shift) noexcept
{
    return (value >> shift) | (value << (32u - shift));
}

class Sha256 {
public:
    void update(const std::uint8_t* data, std::size_t size)
    {
        if (size != 0 && data == nullptr) {
            throw std::invalid_argument("null SHA-256 input");
        }
        if (size > (std::numeric_limits<std::uint64_t>::max() - total_bytes_)) {
            throw std::length_error("SHA-256 input is too large");
        }
        total_bytes_ += size;
        while (size != 0) {
            const std::size_t take = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, take);
            block_size_ += take;
            data += take;
            size -= take;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finish()
    {
        const std::uint64_t bit_count = total_bytes_ * 8u;
        block_[block_size_++] = 0x80u;
        if (block_size_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                      block_.end(), 0u);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                  block_.begin() + 56, 0u);
        for (std::size_t i = 0; i < 8; ++i) {
            block_[63 - i] = static_cast<std::uint8_t>(bit_count >> (8u * i));
        }
        transform(block_.data());

        std::array<std::uint8_t, 32> digest{};
        for (std::size_t i = 0; i < state_.size(); ++i) {
            digest[4 * i] = static_cast<std::uint8_t>(state_[i] >> 24u);
            digest[4 * i + 1] = static_cast<std::uint8_t>(state_[i] >> 16u);
            digest[4 * i + 2] = static_cast<std::uint8_t>(state_[i] >> 8u);
            digest[4 * i + 3] = static_cast<std::uint8_t>(state_[i]);
        }
        return digest;
    }

private:
    void transform(const std::uint8_t* block)
    {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) {
            words[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24u) |
                       (static_cast<std::uint32_t>(block[4 * i + 1]) << 16u) |
                       (static_cast<std::uint32_t>(block[4 * i + 2]) << 8u) |
                       static_cast<std::uint32_t>(block[4 * i + 3]);
        }
        for (std::size_t i = 16; i < words.size(); ++i) {
            const std::uint32_t s0 = rotr(words[i - 15], 7) ^
                                     rotr(words[i - 15], 18) ^
                                     (words[i - 15] >> 3u);
            const std::uint32_t s1 = rotr(words[i - 2], 17) ^
                                     rotr(words[i - 2], 19) ^
                                     (words[i - 2] >> 10u);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t i = 0; i < words.size(); ++i) {
            const std::uint32_t sum1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + sum1 + choose +
                                        kSha256Constants[i] + words[i];
            const std::uint32_t sum0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_{};
    std::uint64_t total_bytes_{};
};

std::string hex_digest(const std::array<std::uint8_t, 32>& digest)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint8_t byte : digest) {
        output << std::setw(2) << static_cast<unsigned>(byte);
    }
    return output.str();
}

} // namespace

std::string sha256_bytes(const std::uint8_t* data, std::size_t size)
{
    Sha256 hash;
    hash.update(data, size);
    return hex_digest(hash.finish());
}

std::string sha256_bytes(const std::vector<std::uint8_t>& data)
{
    return sha256_bytes(data.data(), data.size());
}

std::string sha256_file(const std::filesystem::path& path)
{
    require_absolute_existing_path(path, true);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open file for SHA-256: " + path.string());
    }
    Sha256 hash;
    std::array<char, 1024 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) {
            hash.update(reinterpret_cast<const std::uint8_t*>(buffer.data()),
                        static_cast<std::size_t>(count));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while hashing file: " + path.string());
    }
    return hex_digest(hash.finish());
}

std::string utc_now_iso8601()
{
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto seconds = time_point_cast<std::chrono::seconds>(now);
    const auto micros = duration_cast<microseconds>(now - seconds).count();
    const std::time_t raw = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &raw);
#else
    gmtime_r(&raw, &tm);
#endif
    std::ostringstream output;
    output << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << '.'
           << std::setw(6) << std::setfill('0') << micros << 'Z';
    return output.str();
}

std::filesystem::path require_absolute_existing_path(
    const std::filesystem::path& path,
    bool require_regular_file)
{
    if (!path.is_absolute()) {
        throw std::invalid_argument("path must be absolute: " + path.string());
    }
    if (!std::filesystem::exists(path)) {
        throw std::invalid_argument("path does not exist: " + path.string());
    }
    if (require_regular_file && !std::filesystem::is_regular_file(path)) {
        throw std::invalid_argument("path is not a regular file: " + path.string());
    }
    if (!require_regular_file && !std::filesystem::is_directory(path)) {
        throw std::invalid_argument("path is not a directory: " + path.string());
    }
    return std::filesystem::canonical(path);
}

std::string trial_seed_domain(const TrialDescriptor& descriptor)
{
    if (descriptor.run_id.empty() || descriptor.case_id.empty() ||
        descriptor.purpose.empty()) {
        throw std::invalid_argument("trial seed descriptor contains an empty identity field");
    }
    return "FPT2026/randomness/v1/" + descriptor.run_id + '/' +
           descriptor.case_id + '/' + std::to_string(descriptor.repetition) + '/' +
           std::to_string(descriptor.frame_index) + '/' + descriptor.purpose + '/' +
           std::to_string(descriptor.prime_index);
}

std::array<std::uint8_t, 64> derive_trial_seed(const TrialDescriptor& descriptor)
{
    const auto bytes = shake256(trial_seed_domain(descriptor), 64);
    std::array<std::uint8_t, 64> result{};
    std::copy(bytes.begin(), bytes.end(), result.begin());
    return result;
}

BuildProvenance compiled_build_provenance()
{
    BuildProvenance result;
#if defined(__clang__)
    result.cxx_compiler = std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    result.cxx_compiler = std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    result.cxx_compiler = "msvc " + std::to_string(_MSC_VER);
#else
    result.cxx_compiler = "unknown";
#endif
    result.cxx_standard = std::to_string(__cplusplus);
#ifdef FPT2026_ACCELERATOR_COMMIT
    result.accelerator_commit = FPT2026_ACCELERATOR_COMMIT;
#endif
#ifdef FPT2026_SEAL_EMBEDDED_COMMIT
    result.seal_embedded_commit = FPT2026_SEAL_EMBEDDED_COMMIT;
#endif
#ifdef FPT2026_MICROSOFT_SEAL_COMMIT
    result.microsoft_seal_commit = FPT2026_MICROSOFT_SEAL_COMMIT;
#endif
    return result;
}

} // namespace fpt2026
