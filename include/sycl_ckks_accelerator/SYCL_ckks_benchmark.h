#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sycl_ckks_accelerator/SYCL_ckks_sym.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SYCL_BENCHMARK_ABI_VERSION 1u

typedef struct SYCLBenchmarkSession SYCLBenchmarkSession;

typedef uint32_t SYCLBenchmarkStatus;
enum {
    SYCL_BENCHMARK_STATUS_SUCCESS = 0u,
    SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT = 1u,
    SYCL_BENCHMARK_STATUS_ABI_MISMATCH = 2u,
    SYCL_BENCHMARK_STATUS_LIFECYCLE_ERROR = 3u,
    SYCL_BENCHMARK_STATUS_RUNTIME_ERROR = 4u
};

typedef struct SYCLBenchmarkConfig {
    uint32_t abi_version;
    size_t struct_size;
    size_t n;
    size_t num_moduli;
    size_t max_frames;
    uint32_t save_ntt_s;
    uint32_t save_ntt_pte;
    uint32_t enable_profiling;
} SYCLBenchmarkConfig;

typedef struct SYCLBenchmarkTiming {
    uint32_t abi_version;
    size_t struct_size;
    uint64_t pack_wall_ns;
    uint64_t h2d_wall_ns;
    uint64_t h2d_device_ns;
    uint64_t graph_device_ns;
    uint64_t d2h_device_ns;
    uint64_t d2h_wall_ns;
    uint64_t unpack_wall_ns;
    uint64_t accelerator_api_wall_ns;
    uint64_t first_entry_start_ns;
    uint64_t last_c0_exit_end_ns;
    uint32_t h2d_profiling_available;
    uint32_t graph_profiling_available;
    uint32_t d2h_profiling_available;
    uint32_t additive_wall_breakdown_available;
    uint64_t graph_submit_wait_wall_ns;
} SYCLBenchmarkTiming;

/* Open enums: unknown values must not be treated as exhaustive by callers. */
typedef uint32_t SYCLBenchmarkStage;
enum {
    SYCL_BENCHMARK_STAGE_H2D = 0u,
    SYCL_BENCHMARK_STAGE_ENTRY = 1u,
    SYCL_BENCHMARK_STAGE_IFFT_FANOUT = 2u,
    SYCL_BENCHMARK_STAGE_SCALE_REDUCE = 3u,
    SYCL_BENCHMARK_STAGE_POLY_MULT_NEG_ADD = 4u,
    SYCL_BENCHMARK_STAGE_EXIT_C0 = 5u,
    SYCL_BENCHMARK_STAGE_EXIT_NTT_A = 6u,
    SYCL_BENCHMARK_STAGE_EXIT_NTT_B = 7u,
    SYCL_BENCHMARK_STAGE_D2H = 8u
};

typedef uint32_t SYCLBenchmarkTransferKind;
enum {
    SYCL_BENCHMARK_TRANSFER_NONE = 0u,
    SYCL_BENCHMARK_TRANSFER_PACKED_INPUT = 1u,
    SYCL_BENCHMARK_TRANSFER_C0 = 2u,
    SYCL_BENCHMARK_TRANSFER_NTT_A = 3u,
    SYCL_BENCHMARK_TRANSFER_NTT_B = 4u
};

typedef uint32_t SYCLBenchmarkProfilingUnavailableReason;
enum {
    SYCL_BENCHMARK_PROFILING_AVAILABLE = 0u,
    SYCL_BENCHMARK_PROFILING_DISABLED = 1u,
    SYCL_BENCHMARK_PROFILING_QUERY_FAILED = 2u
};

typedef struct SYCLBenchmarkEventRecord {
    uint32_t abi_version;
    size_t struct_size;
    size_t frame_index;
    int32_t modulus_index;
    SYCLBenchmarkStage stage;
    SYCLBenchmarkTransferKind transfer_kind;
    uint64_t byte_count;
    uint64_t command_start_ns;
    uint64_t command_end_ns;
    uint32_t profiling_available;
    uint32_t unavailable_reason;
} SYCLBenchmarkEventRecord;

/*
 * Creates the process's only benchmark session. The six modulus parameters are
 * retained by the session. required_event_record_capacity receives exactly
 * max_frames * (27 + 12 * save_ntt_s + 12 * save_ntt_pte).
 *
 * The queue and persistent IFFT/NTT services are process-lifetime resources.
 * Close makes the handle terminal but deliberately does not destroy those
 * resources, because queue destruction must not wait for persistent services.
 */
int SYCL_benchmark_session_create(
    const SYCLBenchmarkConfig *config,
    const double *scales,
    const uint32_t *mod_values,
    const uint32_t *const_ratios,
    SYCLBenchmarkSession **session,
    size_t *required_event_record_capacity,
    char *error_message,
    size_t error_message_capacity);

/*
 * All batch arrays are frame-major. encoding_buffers and error_samples contain
 * frame_count * n elements. Each secret_keys[p], uniform_polys[p],
 * c0_outputs[p], c1_outputs[p], and enabled diagnostic output contains
 * frame_count * n elements at frame * n + coefficient. Diagnostic output arrays
 * may be null only when their corresponding config mode is disabled.
 *
 * timing and each record that will be written must be initialized by the caller
 * with this ABI version and its structure size. Capacity failure and all other
 * argument/ABI failures occur before any batch command is submitted. Exactly
 * frame_count * (27 + 12 * save_ntt_s + 12 * save_ntt_pte) records are emitted.
 * For frame_count==1, the additive wall-breakdown flag is true and pack,
 * H2D-wait, graph-submit/wait, D2H-wait, and unpack are disjoint synchronous
 * regions. For larger batches it is false; use the event/device intervals and
 * accelerator_api_wall_ns rather than summing overlapping stages.
 */
int SYCL_benchmark_encrypt_batch(
    SYCLBenchmarkSession *session,
    size_t frame_count,
    const complex_double *encoding_buffers,
    const int8_t *error_samples,
    const uint32_t *const *secret_keys,
    const uint32_t *const *uniform_polys,
    uint32_t **c0_outputs,
    uint32_t **c1_outputs,
    uint32_t **s_save,
    uint32_t **ntt_pte_outputs,
    SYCLBenchmarkTiming *timing,
    SYCLBenchmarkEventRecord *event_records,
    size_t event_record_capacity,
    size_t *event_records_written,
    char *error_message,
    size_t error_message_capacity);

/*
 * Terminal operation. It does not wait on or destroy persistent service work.
 * The handle must not be reused after a successful close.
 */
int SYCL_benchmark_session_close(
    SYCLBenchmarkSession *session,
    char *error_message,
    size_t error_message_capacity);

#ifdef SYCL_CKKS_BENCHMARK_PRIVATE_TEST
/* Private lifecycle seams for host-only FPGA Test sources; not a public ABI. */
typedef void (*SYCLBenchmarkPrivatePostGuardHook)(void *context);

enum {
    SYCL_BENCHMARK_PRIVATE_UNUSED = 0u,
    SYCL_BENCHMARK_PRIVATE_LEGACY_ACTIVE = 1u,
    SYCL_BENCHMARK_PRIVATE_LEGACY_USED = 2u,
    SYCL_BENCHMARK_PRIVATE_SESSION_ACTIVE = 3u,
    SYCL_BENCHMARK_PRIVATE_CLOSED = 4u,
    SYCL_BENCHMARK_PRIVATE_FAILED = 5u
};

void SYCL_benchmark_private_reset_lifecycle(void);
uint32_t SYCL_benchmark_private_lifecycle_state(void);
void SYCL_benchmark_private_set_post_guard_hook(
    SYCLBenchmarkPrivatePostGuardHook hook,
    void *context);
int SYCL_benchmark_private_try_claim_legacy(void);
int SYCL_benchmark_private_try_claim_session(void);
void SYCL_benchmark_private_mark_legacy_used(void);
void SYCL_benchmark_private_mark_session_closed(void);
void SYCL_benchmark_private_mark_session_failed(void);
#endif

#ifdef __cplusplus
}
#endif
