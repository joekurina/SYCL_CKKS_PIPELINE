#include "fpt2026_benchmark/benchmark_record.hpp"

#include "fpt2026_benchmark/benchmark_config.hpp"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

namespace fpt2026 {
namespace {

std::string bool_json(bool value)
{
    return value ? "true" : "false";
}

std::string string_json(const std::string& value)
{
    return '"' + escape_json(value) + '"';
}

std::string optional_string_json(const std::optional<std::string>& value)
{
    return value ? string_json(*value) : "null";
}

template <typename T>
std::string optional_integer_json(const std::optional<T>& value)
{
    return value ? std::to_string(*value) : "null";
}

std::string finite_number_json(double value)
{
    if (!std::isfinite(value)) {
        throw std::invalid_argument("JSON records cannot contain a nonfinite number");
    }
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

std::string optional_number_json(const std::optional<double>& value)
{
    return value ? finite_number_json(*value) : "null";
}

std::string string_array_json(const std::vector<std::string>& values)
{
    std::ostringstream output;
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << string_json(values[i]);
    }
    output << ']';
    return output.str();
}

const char* stage_name(SYCLBenchmarkStage stage)
{
    switch (stage) {
    case SYCL_BENCHMARK_STAGE_H2D: return "H2D";
    case SYCL_BENCHMARK_STAGE_ENTRY: return "ENTRY";
    case SYCL_BENCHMARK_STAGE_IFFT_FANOUT: return "IFFT_FANOUT";
    case SYCL_BENCHMARK_STAGE_SCALE_REDUCE: return "SCALE_REDUCE";
    case SYCL_BENCHMARK_STAGE_POLY_MULT_NEG_ADD: return "POLY_MULT_NEG_ADD";
    case SYCL_BENCHMARK_STAGE_EXIT_C0: return "EXIT_C0";
    case SYCL_BENCHMARK_STAGE_EXIT_NTT_A: return "EXIT_NTT_A";
    case SYCL_BENCHMARK_STAGE_EXIT_NTT_B: return "EXIT_NTT_B";
    case SYCL_BENCHMARK_STAGE_D2H: return "D2H";
    default: throw std::invalid_argument("unknown accelerator event stage");
    }
}

const char* transfer_name(SYCLBenchmarkTransferKind kind)
{
    switch (kind) {
    case SYCL_BENCHMARK_TRANSFER_NONE: return "NONE";
    case SYCL_BENCHMARK_TRANSFER_PACKED_INPUT: return "PACKED_INPUT";
    case SYCL_BENCHMARK_TRANSFER_C0: return "C0";
    case SYCL_BENCHMARK_TRANSFER_NTT_A: return "NTT_A";
    case SYCL_BENCHMARK_TRANSFER_NTT_B: return "NTT_B";
    default: throw std::invalid_argument("unknown accelerator transfer kind");
    }
}

std::string unavailable_reason(std::uint32_t reason)
{
    switch (reason) {
    case SYCL_BENCHMARK_PROFILING_DISABLED: return "profiling_disabled";
    case SYCL_BENCHMARK_PROFILING_QUERY_FAILED: return "profiling_query_failed";
    default: return "provider_did_not_expose_profiling";
    }
}

std::string words_json(const std::array<std::uint64_t, 4>& words)
{
    std::ostringstream output;
    output << '[';
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << words[i];
    }
    output << ']';
    return output.str();
}

std::string moduli_json(const std::array<std::uint64_t, 7>& moduli)
{
    std::ostringstream output;
    output << '[';
    for (std::size_t i = 0; i < moduli.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << moduli[i];
    }
    output << ']';
    return output.str();
}

void write_all(int descriptor, const char* data, std::size_t size)
{
    while (size != 0) {
        const ssize_t written = ::write(descriptor, data, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::string("JSONL write failed: ") + std::strerror(errno));
        }
        if (written == 0) {
            throw std::runtime_error("JSONL write returned zero bytes");
        }
        data += written;
        size -= static_cast<std::size_t>(written);
    }
}

} // namespace

std::string escape_json(std::string_view value)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (byte < 0x20u) {
                output << "\\u" << std::setw(4) << static_cast<unsigned>(byte);
            } else {
                output << static_cast<char>(byte);
            }
        }
    }
    return output.str();
}

std::string timing_record_json(const TimingRecordInput& record)
{
    if (record.identity.run_id.empty() || record.sample_id.empty() ||
        record.batch_size == 0 || record.frame_count_submitted == 0 ||
        record.correctness_record_ids.empty()) {
        throw std::invalid_argument("timing record is missing required identity/count fields");
    }
    std::ostringstream output;
    output << '{'
        << "\"schema_version\":\"1.0\","
        << "\"record_type\":\"timing_sample\","
        << "\"run_id\":" << string_json(record.identity.run_id) << ','
        << "\"sample_id\":" << string_json(record.sample_id) << ','
        << "\"attempt_id\":" << string_json(record.identity.attempt_id) << ','
        << "\"planned_unit_id\":" << string_json(record.identity.planned_unit_id) << ','
        << "\"experiment_id\":" << string_json(record.identity.experiment_id) << ','
        << "\"backend\":" << string_json(record.identity.backend) << ','
        << "\"boundary_id\":" << string_json(kBoundaryId) << ','
        << "\"phase\":" << string_json(record.identity.phase) << ','
        << "\"case_id\":" << string_json(record.identity.case_id) << ','
        << "\"repetition\":" << record.identity.repetition << ','
        << "\"trial_seed_index\":" << record.identity.trial_seed_index << ','
        << "\"batch_size\":" << record.batch_size << ','
        << "\"frame_count_submitted\":" << record.frame_count_submitted << ','
        << "\"frame_count_completed\":" << record.frame_count_completed << ','
        << "\"input\":{"
        << "\"vector_descriptor_sha256\":" << string_json(record.vector_descriptor_sha256) << ','
        << "\"trial_seed_digest\":" << string_json(record.trial_seed_digest) << ','
        << "\"benchmark_key_pair_id\":" << string_json(record.benchmark_key_pair_id) << "},"
        << "\"control_snapshot_ids\":{"
        << "\"before\":" << string_json(record.control_before_id) << ','
        << "\"after\":" << string_json(record.control_after_id) << "},"
        << "\"mode\":{"
        << "\"save_ntt_s\":" << bool_json(record.save_ntt_s) << ','
        << "\"save_ntt_pte\":" << bool_json(record.save_ntt_pte) << "},"
        << "\"timing_ns\":{"
        << "\"application_e2e\":" << record.timing.application_e2e_ns << ','
        << "\"cold_first_result\":" << optional_integer_json(record.cold_first_result_ns) << ','
        << "\"program_time\":" << optional_integer_json(record.program_time_ns) << ','
        << "\"preparation\":" << record.timing.preparation_wall_ns << ','
        << "\"pack\":" << record.timing.pack_wall_ns << ','
        << "\"h2d_wall\":" << record.timing.h2d_wall_ns << ','
        << "\"h2d_device\":"
        << (record.timing.h2d_device_available ? std::to_string(record.timing.h2d_device_ns) : "null") << ','
        << "\"graph_device\":"
        << (record.timing.graph_device_available ? std::to_string(record.timing.graph_device_ns) : "null") << ','
        << "\"d2h_device\":"
        << (record.timing.d2h_device_available ? std::to_string(record.timing.d2h_device_ns) : "null") << ','
        << "\"d2h_wall\":" << record.timing.d2h_wall_ns << ','
        << "\"graph_submit_wait_wall\":" << record.timing.graph_submit_wait_wall_ns << ','
        << "\"unpack_and_assembly\":" << record.timing.unpack_assembly_wall_ns << ','
        << "\"unattributed_wall\":" << record.timing.unattributed_wall_ns << "},"
        << "\"timing_unavailable_reasons\":{";

    bool first_reason = true;
    const auto add_reason = [&](const char* name, bool available, const char* reason) {
        if (!available) {
            if (!first_reason) {
                output << ',';
            }
            output << string_json(name) << ':' << string_json(reason);
            first_reason = false;
        }
    };
    add_reason("cold_first_result", record.cold_first_result_ns.has_value(), "not_this_sample_phase");
    add_reason("program_time", record.program_time_ns.has_value(), "not_this_sample_phase");
    add_reason("h2d_device", record.timing.h2d_device_available, "provider_did_not_expose_copy_profiling");
    add_reason("graph_device", record.timing.graph_device_available, "provider_did_not_expose_graph_profiling");
    add_reason("d2h_device", record.timing.d2h_device_available, "provider_did_not_expose_copy_profiling");

    output << "},"
        << "\"event_record_ids\":" << string_array_json(record.event_record_ids) << ','
        << "\"event_frontiers\":[],"
        << "\"bytes\":{\"h2d\":" << record.h2d_bytes
        << ",\"d2h\":" << record.d2h_bytes << "},"
        << "\"correctness_record_ids\":" << string_array_json(record.correctness_record_ids) << ','
        << "\"correctness_summary\":{"
        << "\"verified_after_timing\":" << bool_json(record.verified_after_timing) << ','
        << "\"max_error\":" << finite_number_json(record.max_error) << ','
        << "\"rms_error\":" << finite_number_json(record.rms_error) << ','
        << "\"mismatch_count\":" << record.mismatch_count << "},"
        << "\"status\":" << string_json(record.status) << ','
        << "\"error\":" << optional_string_json(record.error)
        << '}';
    return output.str();
}

std::string correctness_record_json(const CorrectnessRecordInput& record)
{
    if (record.check_id.empty()) {
        throw std::invalid_argument("correctness record check_id is empty");
    }
    const auto& metrics = record.metrics;
    std::ostringstream output;
    output << '{'
        << "\"schema_version\":\"1.0\","
        << "\"record_type\":\"correctness\","
        << "\"run_id\":" << string_json(record.identity.run_id) << ','
        << "\"correctness_record_id\":" << string_json(record.correctness_record_id) << ','
        << "\"attempt_id\":" << string_json(record.identity.attempt_id) << ','
        << "\"sample_id\":" << optional_string_json(record.sample_id) << ','
        << "\"planned_unit_id\":" << string_json(record.identity.planned_unit_id) << ','
        << "\"experiment_id\":" << string_json(record.identity.experiment_id) << ','
        << "\"backend\":" << string_json(record.identity.backend) << ','
        << "\"case_id\":" << string_json(record.identity.case_id) << ','
        << "\"vector_descriptor_sha256\":" << string_json(record.vector_descriptor_sha256) << ','
        << "\"trial_seed_index\":" << record.identity.trial_seed_index << ','
        << "\"frame_index\":" << record.frame_index << ','
        << "\"check_id\":" << string_json(record.check_id) << ','
        << "\"verification_kind\":\"decrypt_decode\","
        << "\"role\":\"semantic\","
        << "\"selector\":null,\"modulus\":null,\"pattern\":null,"
        << "\"requested_slot_count\":" << metrics.requested_slot_count << ','
        << "\"inactive_slot_count\":" << metrics.inactive_slot_count << ','
        << "\"compared_value_count\":" << metrics.compared_value_count << ','
        << "\"max_abs_error\":" << finite_number_json(metrics.max_abs_error) << ','
        << "\"rms_error\":" << finite_number_json(metrics.rms_error) << ','
        << "\"component_max_abs_error\":" << finite_number_json(metrics.component_max_abs_error) << ','
        << "\"mismatch_count\":" << metrics.mismatch_count << ','
        << "\"threshold\":" << finite_number_json(metrics.threshold) << ','
        << "\"paired_reference_correctness_record_id\":"
        << optional_string_json(record.paired_reference_correctness_record_id) << ','
        << "\"pairwise_max_abs_error\":" << optional_number_json(record.pairwise_max_abs_error) << ','
        << "\"pairwise_rms_error\":" << optional_number_json(record.pairwise_rms_error) << ','
        << "\"passed\":" << bool_json(metrics.passed)
        << '}';
    return output.str();
}

std::string event_record_json(
    const RecordIdentity& identity,
    const std::string& event_record_id,
    const std::optional<std::string>& sample_id,
    const SYCLBenchmarkEventRecord& event)
{
    if (event.abi_version != SYCL_BENCHMARK_ABI_VERSION ||
        event.struct_size < sizeof(event)) {
        throw std::invalid_argument("accelerator event record ABI mismatch");
    }
    const bool available = event.profiling_available != 0;
    if (available && event.command_end_ns < event.command_start_ns) {
        throw std::invalid_argument("accelerator event timestamps are non-monotonic");
    }
    std::ostringstream output;
    output << '{'
        << "\"schema_version\":\"1.0\","
        << "\"record_type\":\"sycl_event\","
        << "\"run_id\":" << string_json(identity.run_id) << ','
        << "\"event_record_id\":" << string_json(event_record_id) << ','
        << "\"attempt_id\":" << string_json(identity.attempt_id) << ','
        << "\"sample_id\":" << optional_string_json(sample_id) << ','
        << "\"phase\":" << string_json(identity.phase) << ','
        << "\"frame_index\":" << event.frame_index << ','
        << "\"stage\":" << string_json(stage_name(event.stage)) << ','
        << "\"transfer_kind\":" << string_json(transfer_name(event.transfer_kind)) << ','
        << "\"modulus_index\":" << event.modulus_index << ','
        << "\"byte_count\":" << event.byte_count << ','
        << "\"profiling_available\":" << bool_json(available) << ','
        << "\"profiling_unavailable_reason\":"
        << (available ? "null" : string_json(unavailable_reason(event.unavailable_reason))) << ','
        << "\"command_start_ns\":"
        << (available ? std::to_string(event.command_start_ns) : "null") << ','
        << "\"command_end_ns\":"
        << (available ? std::to_string(event.command_end_ns) : "null")
        << '}';
    return output.str();
}

std::string context_provenance_json(const ContextProvenance& provenance)
{
    if (provenance.special_prime == 0 || provenance.pipeline_input_block_size == 0 ||
        provenance.uint32_size != 4 || provenance.key_pair_verification.status != "pass" ||
        provenance.key_pair_verification.attempt_id.empty()) {
        throw std::invalid_argument("context provenance is incomplete or unverified");
    }
    std::ostringstream output;
    output << '{'
        << "\"special_prime\":" << provenance.special_prime << ','
        << "\"key_context_moduli\":" << moduli_json(provenance.key_context_moduli) << ','
        << "\"key_parms_id_words\":" << words_json(provenance.key_parms_id_words) << ','
        << "\"first_parms_id_words\":" << words_json(provenance.first_parms_id_words) << ','
        << "\"key_parms_id_bytes_sha256\":" << string_json(provenance.key_parms_id_bytes_sha256) << ','
        << "\"first_parms_id_bytes_sha256\":" << string_json(provenance.first_parms_id_bytes_sha256) << ','
        << "\"ciphertext_size\":" << provenance.ciphertext_size << ','
        << "\"ciphertext_modulus_count\":" << provenance.ciphertext_modulus_count << ','
        << "\"ciphertext_is_ntt_form\":" << bool_json(provenance.ciphertext_is_ntt_form) << ','
        << "\"pipeline_input_block_size\":" << provenance.pipeline_input_block_size << ','
        << "\"uint32_size\":" << provenance.uint32_size << ','
        << "\"key_pair_verification\":{"
        << "\"attempt_id\":" << string_json(provenance.key_pair_verification.attempt_id) << ','
        << "\"status\":\"pass\"}"
        << '}';
    return output.str();
}

void append_jsonl_durable(
    const std::filesystem::path& absolute_path,
    const std::string& json_object)
{
    if (!absolute_path.is_absolute()) {
        throw std::invalid_argument("JSONL output path must be absolute");
    }
    if (json_object.empty() || json_object.front() != '{' || json_object.back() != '}') {
        throw std::invalid_argument("JSONL payload must be one serialized JSON object");
    }
    std::filesystem::create_directories(absolute_path.parent_path());
    const int descriptor = ::open(
        absolute_path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (descriptor < 0) {
        throw std::runtime_error(
            "cannot open JSONL output: " + absolute_path.string() + ": " +
            std::strerror(errno));
    }
    try {
        write_all(descriptor, json_object.data(), json_object.size());
        write_all(descriptor, "\n", 1);
        if (::fsync(descriptor) != 0) {
            throw std::runtime_error(
                std::string("JSONL fsync failed: ") + std::strerror(errno));
        }
        ::close(descriptor);
    } catch (...) {
        ::close(descriptor);
        throw;
    }
}

} // namespace fpt2026
