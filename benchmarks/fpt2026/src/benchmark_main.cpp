#include "fpt2026_benchmark/accelerator_runner.hpp"
#include "fpt2026_benchmark/benchmark_record.hpp"
#include "fpt2026_benchmark/benchmark_vectors.hpp"
#include "fpt2026_benchmark/provenance.hpp"
#include "fpt2026_benchmark/seal_embedded_runner.hpp"
#include "fpt2026_benchmark/seal_oracle.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fpt2026 {
namespace {

class Arguments {
public:
    Arguments(int argc, char** argv)
    {
        if (argc < 2) {
            throw std::invalid_argument("missing benchmark subcommand");
        }
        command_ = argv[1];
        for (int i = 2; i < argc; i += 2) {
            const std::string name = argv[i];
            if (name.rfind("--", 0) != 0 || i + 1 >= argc) {
                throw std::invalid_argument("options must be explicit --name value pairs");
            }
            if (!values_.emplace(name.substr(2), argv[i + 1]).second) {
                throw std::invalid_argument("duplicate option: " + name);
            }
        }
    }

    const std::string& command() const noexcept { return command_; }

    std::string require(const std::string& name) const
    {
        const auto found = values_.find(name);
        if (found == values_.end() || found->second.empty()) {
            throw std::invalid_argument("missing required option --" + name);
        }
        return found->second;
    }

    std::optional<std::string> optional(const std::string& name) const
    {
        const auto found = values_.find(name);
        return found == values_.end() ? std::nullopt
                                      : std::optional<std::string>(found->second);
    }

    std::size_t require_size(const std::string& name) const
    {
        const std::string text = require(name);
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed, 10);
        if (consumed != text.size() || value == 0) {
            throw std::invalid_argument("--" + name + " must be a positive integer");
        }
        return static_cast<std::size_t>(value);
    }

    void reject_unknown(const std::vector<std::string>& allowed) const
    {
        for (const auto& item : values_) {
            if (std::find(allowed.begin(), allowed.end(), item.first) == allowed.end()) {
                throw std::invalid_argument("unknown option --" + item.first);
            }
        }
    }

private:
    std::string command_;
    std::map<std::string, std::string> values_;
};

struct CommonOptions {
    std::string command;
    std::string backend;
    std::string run_id;
    std::string attempt_id;
    std::string planned_unit_id;
    std::string experiment_id;
    std::string benchmark_key_pair_id;
    std::string control_before_id;
    std::string control_after_id;
    std::filesystem::path output;
    std::filesystem::path compact_key;
    std::filesystem::path seal_key;
    std::size_t pipeline_input_block_size{};
};

struct SequenceNumbers {
    std::size_t sample{};
    std::size_t correctness{};
    std::size_t event{};
    std::size_t frame{};
};

CommonOptions parse_common(const Arguments& arguments)
{
    CommonOptions options;
    options.command = arguments.command();
    options.backend = arguments.require("backend");
    if (options.backend != "fpga" && options.backend != "seal-embedded" &&
        options.backend != "stock-seal-reference") {
        throw std::invalid_argument("unknown backend: " + options.backend);
    }
    options.run_id = arguments.require("run-id");
    options.attempt_id = arguments.require("attempt-id");
    options.planned_unit_id = arguments.require("planned-unit-id");
    options.experiment_id = arguments.require("experiment-id");
    options.benchmark_key_pair_id = arguments.require("benchmark-key-pair-id");
    options.control_before_id = arguments.require("control-before-id");
    options.control_after_id = arguments.require("control-after-id");
    options.output = arguments.require("output");
    options.compact_key = arguments.require("compact-key");
    options.seal_key = arguments.require("seal-key");
    options.pipeline_input_block_size = arguments.require_size("pipeline-input-block-size");

    if (options.experiment_id.size() != 2 || options.experiment_id[0] != 'E' ||
        options.experiment_id[1] < '1' || options.experiment_id[1] > '8') {
        throw std::invalid_argument("--experiment-id must be E1 through E8");
    }
    options.output = std::filesystem::absolute(options.output);
    if (!options.output.is_absolute() || !options.compact_key.is_absolute() ||
        !options.seal_key.is_absolute()) {
        throw std::invalid_argument("output and key paths must be absolute");
    }
    std::filesystem::create_directories(options.output);
    return options;
}

std::vector<std::size_t> parse_size_list(const std::string& text)
{
    std::vector<std::size_t> result;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::string item = text.substr(
            begin, comma == std::string::npos ? std::string::npos : comma - begin);
        std::size_t consumed = 0;
        const auto value = std::stoull(item, &consumed, 10);
        if (consumed != item.size() || value == 0) {
            throw std::invalid_argument("batch list contains a non-positive integer");
        }
        result.push_back(static_cast<std::size_t>(value));
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    if (result.empty()) {
        throw std::invalid_argument("batch list is empty");
    }
    return result;
}

std::vector<std::size_t> parse_index_list(const std::string& text)
{
    std::vector<std::size_t> result;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::string item = text.substr(
            begin, comma == std::string::npos ? std::string::npos : comma - begin);
        std::size_t consumed = 0;
        const auto value = std::stoull(item, &consumed, 10);
        if (item.empty() || consumed != item.size()) {
            throw std::invalid_argument("index list contains a non-integer");
        }
        result.push_back(static_cast<std::size_t>(value));
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    if (result.empty() ||
        std::adjacent_find(result.begin(), result.end()) != result.end()) {
        throw std::invalid_argument("index list is empty or contains adjacent duplicates");
    }
    return result;
}

std::string vector_digest(const BenchmarkVector& input)
{
    return sha256_bytes(serialize_vector_artifact(input));
}

std::string aggregate_digest(const std::vector<std::string>& digests)
{
    std::string joined;
    for (const auto& digest : digests) {
        joined += digest;
        joined.push_back('\n');
    }
    return sha256_bytes(reinterpret_cast<const std::uint8_t*>(joined.data()), joined.size());
}

void publish_vector_artifact(
    const std::filesystem::path& output,
    const BenchmarkVector& vector)
{
    const auto artifact = serialize_vector_artifact(vector);
    const auto path = output / (vector.id + ".fptvec");
    if (std::filesystem::exists(path)) {
        if (sha256_file(path) != sha256_bytes(artifact)) {
            throw std::runtime_error("existing vector artifact has different contents: " + path.string());
        }
        return;
    }
    std::ofstream stream(path, std::ios::binary | std::ios::out | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(artifact.data()),
                 static_cast<std::streamsize>(artifact.size()));
    stream.flush();
    if (!stream) {
        throw std::runtime_error("failed to publish vector artifact: " + path.string());
    }
}

std::vector<TrialSeeds> make_seeds(
    const CommonOptions& options,
    const std::vector<BenchmarkVector>& frames,
    std::size_t repetition,
    std::size_t first_frame_index)
{
    std::vector<TrialSeeds> seeds;
    seeds.reserve(frames.size());
    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        seeds.push_back(make_trial_seeds(
            options.run_id, frames[frame].id, repetition, first_frame_index + frame));
    }
    return seeds;
}

RecordIdentity identity_for(
    const CommonOptions& options,
    std::string phase,
    std::string case_id,
    std::size_t repetition)
{
    return RecordIdentity{
        options.run_id,
        options.attempt_id,
        options.planned_unit_id,
        options.experiment_id,
        options.backend,
        std::move(phase),
        std::move(case_id),
        repetition,
        repetition,
    };
}

std::vector<EventFrontierRecord> derive_event_frontiers(
    const std::vector<SYCLBenchmarkEventRecord>& events,
    std::size_t frame_count)
{
    std::vector<EventFrontierRecord> frontiers(frame_count);
    std::vector<std::array<std::size_t, 5>> counts(frame_count);
    std::vector<bool> all_available(frame_count, true);
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        frontiers[frame].frame_index = frame;
    }
    const auto update_max = [](std::optional<std::uint64_t>& value, std::uint64_t candidate) {
        value = value ? std::max(*value, candidate) : candidate;
    };
    for (const auto& event : events) {
        if (event.frame_index >= frame_count) {
            throw std::runtime_error("event frame index is outside the submitted batch");
        }
        std::size_t dimension = 5;
        std::optional<std::uint64_t>* destination = nullptr;
        auto& frontier = frontiers[event.frame_index];
        switch (event.stage) {
        case SYCL_BENCHMARK_STAGE_ENTRY:
            dimension = 0;
            destination = &frontier.entry_end_ns;
            break;
        case SYCL_BENCHMARK_STAGE_IFFT_FANOUT:
            dimension = 1;
            destination = &frontier.fanout_end_ns;
            break;
        case SYCL_BENCHMARK_STAGE_SCALE_REDUCE:
            dimension = 2;
            destination = &frontier.scale_end_ns;
            break;
        case SYCL_BENCHMARK_STAGE_POLY_MULT_NEG_ADD:
            dimension = 3;
            destination = &frontier.poly_end_ns;
            break;
        case SYCL_BENCHMARK_STAGE_EXIT_C0:
            dimension = 4;
            destination = &frontier.exit_end_ns;
            break;
        default:
            break;
        }
        if (destination == nullptr) {
            continue;
        }
        ++counts[event.frame_index][dimension];
        if (event.profiling_available == 0) {
            all_available[event.frame_index] = false;
        } else {
            update_max(*destination, event.command_end_ns);
        }
    }
    constexpr std::array<std::size_t, 5> expected_counts = {1, 1, 6, 6, 6};
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        if (counts[frame] != expected_counts) {
            throw std::runtime_error("bounded event cardinality cannot produce frame frontiers");
        }
        auto& frontier = frontiers[frame];
        if (!all_available[frame]) {
            frontier.entry_end_ns.reset();
            frontier.fanout_end_ns.reset();
            frontier.scale_end_ns.reset();
            frontier.poly_end_ns.reset();
            frontier.exit_end_ns.reset();
            frontier.unavailable_reason = "one_or_more_bounded_events_lack_profiling";
        } else {
            frontier.profiling_available = true;
        }
    }
    return frontiers;
}

void emit_batch(
    const CommonOptions& options,
    const std::vector<BenchmarkVector>& frames,
    const BackendBatchResult& result,
    const std::vector<SYCLBenchmarkEventRecord>* events,
    const std::string& phase,
    const std::string& check_id,
    std::size_t repetition,
    bool measured,
    SequenceNumbers& sequence,
    std::optional<std::uint64_t> cold_first_result_ns = std::nullopt)
{
    if (frames.size() != result.correctness.size() ||
        frames.size() != result.trial_seed_digests.size()) {
        throw std::runtime_error("backend result count does not match submitted frames");
    }
    const std::string sample_id = options.attempt_id + "-sample-" +
                                  std::to_string(sequence.sample);
    const std::size_t frame_base = sequence.frame;
    std::vector<std::string> correctness_ids;
    correctness_ids.reserve(frames.size());
    std::vector<std::string> vector_digests;
    vector_digests.reserve(frames.size());

    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        const std::string correctness_id = options.attempt_id + "-correctness-" +
                                           std::to_string(sequence.correctness++);
        correctness_ids.push_back(correctness_id);
        vector_digests.push_back(vector_digest(frames[frame]));
        CorrectnessRecordInput record;
        record.identity = identity_for(options, phase, frames[frame].id, repetition);
        record.correctness_record_id = correctness_id;
        record.check_id = check_id;
        if (measured) {
            record.sample_id = sample_id;
        }
        record.vector_descriptor_sha256 = vector_digests.back();
        record.trial_seed_digest = result.trial_seed_digests[frame];
        record.benchmark_key_pair_id = options.benchmark_key_pair_id;
        record.oracle_id = "stock-seal-decrypt-decode-v1";
        record.verified_utc = utc_now_iso8601();
        record.frame_index = frame_base + frame;
        record.metrics = result.correctness[frame];
        if (!result.transport_correctness.empty()) {
            if (result.transport_correctness.size() != frames.size()) {
                throw std::runtime_error("transport correctness count does not match frames");
            }
            record.transport_metrics = result.transport_correctness[frame];
        }
        append_jsonl_durable(
            options.output / "correctness.jsonl",
            correctness_record_json(record));
    }

    std::vector<std::string> event_ids;
    std::uint64_t h2d_bytes = 0;
    std::uint64_t d2h_bytes = 0;
    if (events != nullptr) {
        event_ids.reserve(events->size());
        for (const auto& event : *events) {
            SYCLBenchmarkEventRecord emitted_event = event;
            emitted_event.frame_index += frame_base;
            const std::string event_id = options.attempt_id + "-event-" +
                                         std::to_string(sequence.event++);
            event_ids.push_back(event_id);
            if (event.stage == SYCL_BENCHMARK_STAGE_H2D) {
                h2d_bytes += event.byte_count;
            }
            if (event.stage == SYCL_BENCHMARK_STAGE_D2H) {
                d2h_bytes += event.byte_count;
            }
            auto event_identity = identity_for(
                options, phase,
                event.frame_index < frames.size() ? frames[event.frame_index].id : frames.front().id,
                repetition);
            append_jsonl_durable(
                options.output / "events.jsonl",
                event_record_json(
                    event_identity, event_id,
                    measured ? std::optional<std::string>(sample_id) : std::nullopt,
                    emitted_event));
        }
    }

    std::vector<EventFrontierRecord> event_frontiers;
    if (events != nullptr) {
        event_frontiers = derive_event_frontiers(*events, frames.size());
        for (auto& frontier : event_frontiers) {
            frontier.frame_index += frame_base;
        }
    }
    sequence.frame += frames.size();

    if (!measured) {
        return;
    }

    double max_error = 0.0;
    long double rms_square_sum = 0.0L;
    std::size_t mismatch_count = 0;
    for (const auto& metrics : result.correctness) {
        max_error = std::max(max_error, metrics.max_abs_error);
        rms_square_sum += static_cast<long double>(metrics.rms_error) * metrics.rms_error;
        mismatch_count += metrics.mismatch_count;
    }

    TimingRecordInput sample;
    sample.identity = identity_for(options, phase, frames.front().id, repetition);
    sample.sample_id = sample_id;
    sample.vector_descriptor_sha256 = aggregate_digest(vector_digests);
    sample.trial_seed_digest = aggregate_digest(result.trial_seed_digests);
    sample.benchmark_key_pair_id = options.benchmark_key_pair_id;
    sample.control_before_id = options.control_before_id;
    sample.control_after_id = options.control_after_id;
    sample.batch_size = frames.size();
    sample.frame_count_submitted = frames.size();
    sample.frame_count_completed = result.correctness.size();
    sample.timing = result.timing;
    const bool publish_additive_wall =
        sample.timing.additive_wall_breakdown_available &&
        (options.experiment_id == "E2" || options.experiment_id == "E6");
    if (sample.timing.additive_wall_breakdown_available && !publish_additive_wall) {
        sample.timing.unattributed_wall_ns += sample.timing.h2d_wall_ns +
            sample.timing.graph_submit_wait_wall_ns + sample.timing.d2h_wall_ns;
        sample.timing.additive_wall_breakdown_available = false;
    }
    sample.cold_first_result_ns = cold_first_result_ns;
    sample.h2d_bytes = h2d_bytes;
    sample.d2h_bytes = d2h_bytes;
    sample.event_record_ids = std::move(event_ids);
    sample.event_frontiers = std::move(event_frontiers);
    sample.correctness_record_ids = std::move(correctness_ids);
    sample.max_error = max_error;
    sample.rms_error = std::sqrt(
        static_cast<double>(rms_square_sum / result.correctness.size()));
    sample.mismatch_count = mismatch_count;
    sample.verified_after_timing = true;
    sample.status = result.passed ? "pass" : "fail";
    if (!result.passed) {
        sample.error = "post-timing correctness verification failed";
    }
    append_jsonl_durable(options.output / "samples.jsonl", timing_record_json(sample));
    ++sequence.sample;
}

BackendBatchResult stock_reference_batch(
    StockSealOracle& oracle,
    const std::vector<BenchmarkVector>& frames,
    const std::vector<TrialSeeds>& seeds)
{
    if (frames.empty() || frames.size() != seeds.size()) {
        throw std::invalid_argument("stock reference frame and seed counts differ");
    }
    BackendBatchResult result;
    result.residues.frame_count = frames.size();
    result.passed = true;
    result.ciphertexts.reserve(frames.size());
    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        result.trial_seed_digests.push_back(seeds[frame].digest);
        auto ciphertext = oracle.encrypt_reference(frames[frame].slots);
        auto decoded = oracle.decrypt_decode(ciphertext);
        auto metrics = evaluate_slots(
            frames[frame].slots, decoded, frames[frame].active_slots,
            kCorrectnessThreshold);
        result.passed = result.passed && metrics.passed;
        result.ciphertexts.push_back(std::move(ciphertext));
        result.decoded_slots.push_back(std::move(decoded));
        result.correctness.push_back(std::move(metrics));
    }
    return result;
}

std::vector<BenchmarkVector> repeat_vector(const BenchmarkVector& input, std::size_t count)
{
    return std::vector<BenchmarkVector>(count, input);
}

std::vector<std::string> common_allowed()
{
    return {
        "backend", "run-id", "attempt-id", "planned-unit-id", "experiment-id",
        "benchmark-key-pair-id", "control-before-id", "control-after-id", "output",
        "compact-key", "seal-key", "pipeline-input-block-size",
    };
}

void add_allowed(std::vector<std::string>& values, std::initializer_list<const char*> names)
{
    for (const char* name : names) {
        values.emplace_back(name);
    }
}

} // namespace
} // namespace fpt2026

int main(int argc, char** argv)
{
    using namespace fpt2026;
    const auto process_begin = std::chrono::steady_clock::now();
    try {
        Arguments arguments(argc, argv);
        const CommonOptions options = parse_common(arguments);
        auto allowed = common_allowed();
        if (options.command == "correctness") {
            add_allowed(allowed, {"profile", "suite", "save-mode", "trial-seeds"});
            if (arguments.require("profile") != "paper") {
                throw std::invalid_argument("correctness supports only --profile paper");
            }
        } else if (options.command == "latency") {
            add_allowed(allowed, {"case", "warmup", "repetitions"});
        } else if (options.command == "throughput") {
            add_allowed(allowed, {"case", "batches", "repetitions"});
        } else if (options.command == "robustness") {
            add_allowed(allowed, {"frames", "batch"});
        } else if (options.command == "cold-start") {
            add_allowed(allowed, {"case"});
        } else if (options.command == "profiler-diagnostic") {
            add_allowed(allowed, {"case", "batch"});
        } else {
            throw std::invalid_argument("unknown benchmark subcommand: " + options.command);
        }
        arguments.reject_unknown(allowed);

        StockSealOracle oracle(options.seal_key);
        const bool key_pair_verified = oracle.verify_compact_secret_key(options.compact_key);
        if (!key_pair_verified) {
            throw std::runtime_error("compact and serialized benchmark keys do not match");
        }
        SealEmbeddedRunner cpu(options.compact_key, oracle);
        const auto context = oracle.context_provenance(
            options.pipeline_input_block_size, options.attempt_id, key_pair_verified);
        append_jsonl_durable(
            options.output / "context-provenance.jsonl",
            context_provenance_json(context));

        SequenceNumbers sequence;
        bool all_passed = true;

        if (options.command == "correctness") {
            const std::string suite = arguments.require("suite");
            const std::string save_mode = arguments.require("save-mode");
            if (save_mode != "performance" && save_mode != "full") {
                throw std::invalid_argument("--save-mode must be performance or full");
            }
            if (suite == "c1-c2") {
                throw std::runtime_error(
                    "C1/C2 suite requires the independent production-path NTT oracle");
            }
            if (suite != "fpga-test" && suite != "standalone-semantic" &&
                suite != "semantic-matrix") {
                throw std::invalid_argument("unknown correctness suite: " + suite);
            }
            if ((suite == "fpga-test" || suite == "standalone-semantic") &&
                options.backend != "fpga") {
                throw std::invalid_argument("E1 correctness suites require --backend fpga");
            }
            std::vector<BenchmarkVector> frames;
            for (const auto id : all_vector_cases()) {
                if (suite == "fpga-test" && is_complex_case(id)) {
                    continue;
                }
                if (options.backend == "seal-embedded" && is_complex_case(id)) {
                    continue;
                }
                if (options.backend == "stock-seal-reference" && !is_complex_case(id)) {
                    continue;
                }
                frames.push_back(generate_benchmark_vector(id));
                publish_vector_artifact(options.output, frames.back());
            }
            const std::vector<std::size_t> trial_indices = suite == "semantic-matrix"
                ? parse_index_list(arguments.require("trial-seeds"))
                : std::vector<std::size_t>{0};
            if (suite == "semantic-matrix" &&
                trial_indices != std::vector<std::size_t>{0, 1, 2, 3, 4}) {
                throw std::invalid_argument(
                    "semantic-matrix requires exact trial seeds 0,1,2,3,4");
            }
            std::optional<AcceleratorRunner> accelerator;
            if (options.backend == "fpga") {
                const bool full_diagnostics = save_mode == "full";
                accelerator.emplace(
                    cpu, oracle, frames.size(), full_diagnostics, full_diagnostics);
            }
            for (const std::size_t trial_index : trial_indices) {
                const auto seeds = make_seeds(options, frames, trial_index, sequence.frame);
                const std::string check_id = suite == "fpga-test" ? "FPGA-Test" : "C3";
                if (accelerator) {
                    const auto result = accelerator->encrypt(frames, seeds);
                    if (result.pipeline_input_block_size != options.pipeline_input_block_size) {
                        throw std::runtime_error(
                            "measured PipelineInputBlock size differs from provenance");
                    }
                    emit_batch(options, frames, result, &result.events, "correctness", check_id,
                               trial_index, false, sequence);
                    all_passed = all_passed && result.passed;
                } else if (options.backend == "seal-embedded") {
                    const auto result = cpu.encrypt(frames, seeds);
                    emit_batch(options, frames, result, nullptr, "correctness", check_id,
                               trial_index, false, sequence);
                    all_passed = all_passed && result.passed;
                } else {
                    const auto result = stock_reference_batch(oracle, frames, seeds);
                    emit_batch(options, frames, result, nullptr, "correctness", check_id,
                               trial_index, false, sequence);
                    all_passed = all_passed && result.passed;
                }
            }
            if (accelerator) {
                accelerator->close();
            }
        } else if (options.command == "latency") {
            if (options.backend == "stock-seal-reference") {
                throw std::invalid_argument("stock-seal-reference is not an E6 timing backend");
            }
            const auto input = generate_benchmark_vector(parse_vector_case(arguments.require("case")));
            publish_vector_artifact(options.output, input);
            const std::size_t warmups = arguments.require_size("warmup");
            const std::size_t repetitions = arguments.require_size("repetitions");
            std::optional<AcceleratorRunner> accelerator;
            if (options.backend == "fpga") {
                accelerator.emplace(cpu, oracle, 1, false, false);
            }
            for (std::size_t repetition = 0; repetition < warmups + repetitions; ++repetition) {
                const auto frames = repeat_vector(input, 1);
                const auto seeds = make_seeds(options, frames, repetition, sequence.frame);
                const bool measured = repetition >= warmups;
                const std::string phase = measured ? "warm_measured" : "warmup";
                if (accelerator) {
                    const auto result = accelerator->encrypt(frames, seeds);
                    if (result.pipeline_input_block_size != options.pipeline_input_block_size) {
                        throw std::runtime_error("measured PipelineInputBlock size differs from provenance");
                    }
                    emit_batch(options, frames, result, &result.events, phase,
                               measured ? "timed-semantic" : "semantic",
                               repetition, measured, sequence);
                    all_passed = all_passed && result.passed;
                } else {
                    const auto result = cpu.encrypt(frames, seeds);
                    emit_batch(options, frames, result, nullptr, phase,
                               measured ? "timed-semantic" : "semantic",
                               repetition, measured, sequence);
                    all_passed = all_passed && result.passed;
                }
            }
            if (accelerator) {
                accelerator->close();
            }
        } else if (options.command == "throughput") {
            if (options.backend != "fpga") {
                throw std::invalid_argument("throughput requires --backend fpga");
            }
            const auto input = generate_benchmark_vector(parse_vector_case(arguments.require("case")));
            publish_vector_artifact(options.output, input);
            const auto batches = parse_size_list(arguments.require("batches"));
            const std::size_t repetitions = arguments.require_size("repetitions");
            const std::size_t maximum = *std::max_element(batches.begin(), batches.end());
            AcceleratorRunner accelerator(cpu, oracle, maximum, false, false);
            std::size_t frame_index = 0;
            for (const std::size_t batch : batches) {
                const auto warm_frames = repeat_vector(input, batch);
                auto warm_seeds = make_seeds(options, warm_frames, 0, frame_index);
                const auto warm = accelerator.encrypt(warm_frames, warm_seeds);
                emit_batch(options, warm_frames, warm, &warm.events, "warmup", "semantic", 0,
                           false, sequence);
                all_passed = all_passed && warm.passed;
                frame_index += batch;
            }
            for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
                for (std::size_t position = 0; position < batches.size(); ++position) {
                    const std::size_t batch = batches[(repetition + position) % batches.size()];
                    const auto frames = repeat_vector(input, batch);
                    const auto seeds = make_seeds(options, frames, repetition, frame_index);
                    const auto result = accelerator.encrypt(frames, seeds);
                    if (result.pipeline_input_block_size != options.pipeline_input_block_size) {
                        throw std::runtime_error("measured PipelineInputBlock size differs from provenance");
                    }
                    emit_batch(options, frames, result, &result.events, "warm_measured",
                               "timed-semantic", repetition, true, sequence);
                    all_passed = all_passed && result.passed;
                    frame_index += batch;
                }
            }
            accelerator.close();
        } else if (options.command == "robustness") {
            if (options.backend != "fpga") {
                throw std::invalid_argument("robustness requires --backend fpga");
            }
            const std::size_t total_frames = arguments.require_size("frames");
            const std::size_t batch_size = arguments.require_size("batch");
            const std::vector<VectorCaseId> rotation{
                VectorCaseId::real_impulse,
                VectorCaseId::real_short_mixed,
                VectorCaseId::real_partial_64,
                VectorCaseId::real_full_4096,
            };
            AcceleratorRunner accelerator(cpu, oracle, batch_size, false, false);
            std::size_t completed = 0;
            std::size_t repetition = 0;
            while (completed < total_frames) {
                const std::size_t count = std::min(batch_size, total_frames - completed);
                std::vector<BenchmarkVector> frames;
                frames.reserve(count);
                for (std::size_t i = 0; i < count; ++i) {
                    frames.push_back(generate_benchmark_vector(rotation[(completed + i) % rotation.size()]));
                    publish_vector_artifact(options.output, frames.back());
                }
                const auto seeds = make_seeds(options, frames, repetition, completed);
                const auto result = accelerator.encrypt(frames, seeds);
                emit_batch(options, frames, result, &result.events, "resident_measured",
                           "resident-semantic", repetition, true, sequence);
                all_passed = all_passed && result.passed;
                completed += count;
                ++repetition;
            }
            accelerator.close();
        } else if (options.command == "cold-start") {
            if (options.backend != "fpga") {
                throw std::invalid_argument("cold-start requires --backend fpga");
            }
            const auto input = generate_benchmark_vector(
                parse_vector_case(arguments.require("case")));
            publish_vector_artifact(options.output, input);
            AcceleratorRunner accelerator(cpu, oracle, 1, false, false);
            for (std::size_t repetition = 0; repetition < 2; ++repetition) {
                const auto frames = repeat_vector(input, 1);
                const auto seeds = make_seeds(options, frames, repetition, sequence.frame);
                const auto result = accelerator.encrypt(frames, seeds);
                const auto verified = std::chrono::steady_clock::now();
                const auto cold_ns = repetition == 0
                    ? std::optional<std::uint64_t>(static_cast<std::uint64_t>(
                          std::chrono::duration_cast<std::chrono::nanoseconds>(
                              verified - process_begin).count()))
                    : std::nullopt;
                emit_batch(options, frames, result, &result.events,
                           repetition == 0 ? "cold_first_result" : "warm_after_first",
                           "timed-semantic", repetition, true, sequence, cold_ns);
                all_passed = all_passed && result.passed;
            }
            accelerator.close();
        } else {
            const auto input = generate_benchmark_vector(parse_vector_case(arguments.require("case")));
            publish_vector_artifact(options.output, input);
            const std::size_t batch = arguments.require_size("batch");
            const auto frames = repeat_vector(input, batch);
            const auto seeds = make_seeds(options, frames, 0, sequence.frame);
            if (options.backend == "fpga") {
                AcceleratorRunner accelerator(cpu, oracle, batch, false, false);
                const auto result = accelerator.encrypt(frames, seeds);
                emit_batch(options, frames, result, &result.events,
                           "diagnostic", "timed-semantic", 0, true, sequence);
                all_passed = result.passed;
                accelerator.close();
            } else if (options.backend == "seal-embedded") {
                const auto result = cpu.encrypt(frames, seeds);
                emit_batch(options, frames, result, nullptr,
                           "diagnostic", "timed-semantic", 0, true, sequence);
                all_passed = result.passed;
            } else {
                throw std::invalid_argument("stock-seal-reference is correctness-only");
            }
        }

        if (!all_passed) {
            std::cerr << "FPT 2026 benchmark correctness failure\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fpt2026_benchmark: " << error.what() << '\n';
        return 2;
    }
}
