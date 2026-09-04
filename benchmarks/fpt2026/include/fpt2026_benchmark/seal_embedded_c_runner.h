#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FPT2026_SEAL_EMBEDDED_C_ABI_VERSION 1u
#define FPT2026_SEAL_EMBEDDED_SEED_BYTES 64u
#define FPT2026_SEAL_EMBEDDED_DEGREE 8192u
#define FPT2026_SEAL_EMBEDDED_MODULI 6u

typedef struct FPT2026SealEmbeddedState FPT2026SealEmbeddedState;

typedef struct FPT2026SealEmbeddedConfig {
    uint32_t abi_version;
    size_t struct_size;
    size_t degree;
    size_t num_moduli;
    double scale;
    const char *compact_key_path;
} FPT2026SealEmbeddedConfig;

typedef struct FPT2026SealEmbeddedRunResult {
    uint32_t abi_version;
    size_t struct_size;
    size_t modulus_visit_count;
    uint32_t modulus_visits[FPT2026_SEAL_EMBEDDED_MODULI];
} FPT2026SealEmbeddedRunResult;

int fpt2026_seal_embedded_create(
    const FPT2026SealEmbeddedConfig *config,
    FPT2026SealEmbeddedState **state,
    char *error_message,
    size_t error_message_capacity);

int fpt2026_seal_embedded_copy_parameters(
    const FPT2026SealEmbeddedState *state,
    uint32_t *moduli,
    size_t moduli_capacity,
    uint32_t *const_ratios,
    size_t ratio_capacity,
    char *error_message,
    size_t error_message_capacity);

int fpt2026_seal_embedded_copy_index_map(
    const FPT2026SealEmbeddedState *state,
    uint16_t *index_map,
    size_t index_map_capacity,
    char *error_message,
    size_t error_message_capacity);

/*
 * Executes the production sequence exactly once:
 * reset, encode_base, sym_init, then six encode_encrypt calls with next-prime
 * calls only between moduli. values contains exactly degree/2 real slots.
 * c0/c1 use modulus-major layout [p * degree + coefficient].
 */
int fpt2026_seal_embedded_encrypt(
    FPT2026SealEmbeddedState *state,
    const float *values,
    size_t value_count,
    const uint8_t shareable_seed[FPT2026_SEAL_EMBEDDED_SEED_BYTES],
    const uint8_t error_seed[FPT2026_SEAL_EMBEDDED_SEED_BYTES],
    uint32_t *c0,
    size_t c0_capacity,
    uint32_t *c1,
    size_t c1_capacity,
    FPT2026SealEmbeddedRunResult *result,
    char *error_message,
    size_t error_message_capacity);

/*
 * Uses SEAL-Embedded's production samplers and loaded compact key to prepare the
 * exact accelerator inputs for a batch. error_samples is frame-major. Secret
 * keys and uniform polynomials are modulus-major, then frame-major.
 */
int fpt2026_seal_embedded_prepare_accelerator_inputs(
    FPT2026SealEmbeddedState *state,
    size_t frame_count,
    const uint8_t *shareable_seeds,
    size_t shareable_seed_bytes,
    const uint8_t *error_seeds,
    size_t error_seed_bytes,
    int8_t *error_samples,
    size_t error_capacity,
    uint32_t *secret_keys,
    size_t secret_key_capacity,
    uint32_t *uniform_polys,
    size_t uniform_capacity,
    char *error_message,
    size_t error_message_capacity);

void fpt2026_seal_embedded_destroy(FPT2026SealEmbeddedState *state);

#ifdef __cplusplus
}
#endif
