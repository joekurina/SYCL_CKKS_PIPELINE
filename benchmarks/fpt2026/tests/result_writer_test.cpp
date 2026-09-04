#include "fpt2026_benchmark/benchmark_record.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    using namespace fpt2026;

    require(escape_json("a\n\"b\\c") == "a\\n\\\"b\\\\c",
            "JSON escaping mismatch");

    ContextProvenance context;
    context.special_prime = 123;
    context.key_context_moduli = {1, 2, 3, 4, 5, 6, 123};
    context.key_parms_id_words = {7, 8, 9, 10};
    context.first_parms_id_words = {11, 12, 13, 14};
    context.key_parms_id_bytes_sha256 = std::string(64, 'a');
    context.first_parms_id_bytes_sha256 = std::string(64, 'b');
    context.pipeline_input_block_size = 260;
    context.key_pair_verification = {"key-check-1", "pass"};
    const auto context_json = context_provenance_json(context);
    for (const char* field : {
             "\"special_prime\"", "\"key_context_moduli\"",
             "\"key_parms_id_words\"", "\"first_parms_id_words\"",
             "\"key_parms_id_bytes_sha256\"", "\"first_parms_id_bytes_sha256\"",
             "\"ciphertext_size\":2", "\"ciphertext_modulus_count\":6",
             "\"ciphertext_is_ntt_form\":true", "\"pipeline_input_block_size\":260",
             "\"uint32_size\":4", "\"key_pair_verification\""}) {
        require(context_json.find(field) != std::string::npos,
                "context provenance field is missing");
    }

    CorrectnessRecordInput correctness;
    correctness.identity = {"run", "attempt", "unit", "E5", "fpga",
                            "measured", "real_impulse", 0, 0};
    correctness.correctness_record_id = "correctness-1";
    correctness.check_id = "timed-semantic";
    correctness.sample_id = "sample-1";
    correctness.vector_descriptor_sha256 = std::string(64, 'c');
    correctness.metrics.requested_slot_count = 1;
    correctness.metrics.inactive_slot_count = 4095;
    correctness.metrics.compared_value_count = 4096;
    correctness.metrics.threshold = 0.1;
    correctness.metrics.passed = true;
    const auto correctness_json = correctness_record_json(correctness);
    require(correctness_json.find("\"check_id\":\"timed-semantic\"") != std::string::npos,
            "correctness check_id is missing");

    TimingRecordInput timing;
    timing.identity = correctness.identity;
    timing.sample_id = "sample-1";
    timing.vector_descriptor_sha256 = std::string(64, 'c');
    timing.trial_seed_digest = std::string(64, 'd');
    timing.benchmark_key_pair_id = "key-pair";
    timing.control_before_id = "before";
    timing.control_after_id = "after";
    timing.batch_size = 1;
    timing.frame_count_submitted = 1;
    timing.frame_count_completed = 1;
    timing.correctness_record_ids = {"correctness-1"};
    timing.verified_after_timing = true;
    timing.status = "pass";
    const auto timing_json = timing_record_json(timing);
    require(timing_json.find("\"h2d_device\":null") != std::string::npos &&
                timing_json.find("provider_did_not_expose_copy_profiling") != std::string::npos,
            "unavailable device timing lacks null/reason pairing");

    std::cout << "result_writer_test: PASS\n";
    return 0;
}
