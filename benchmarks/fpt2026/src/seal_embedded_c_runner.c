#include "fpt2026_benchmark/seal_embedded_c_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckks_common.h"
#include "ckks_sym.h"
#include "parameters.h"
#include "sample.h"

#ifndef SE_USE_MALLOC
#error "The FPT 2026 CPU shim requires the SE_USE_MALLOC SEAL-Embedded configuration"
#endif

_Static_assert(sizeof(ZZ) == sizeof(uint32_t),
               "FPT 2026 requires the 32-bit SEAL-Embedded residue ABI");
_Static_assert(sizeof(flpt) == sizeof(float),
               "FPT 2026 requires the 32-bit SEAL-Embedded real input ABI");

enum {
    FPT2026_SE_OK = 0,
    FPT2026_SE_INVALID_ARGUMENT = 1,
    FPT2026_SE_ABI_MISMATCH = 2,
    FPT2026_SE_SETUP_ERROR = 3,
    FPT2026_SE_ENCODE_ERROR = 4,
    FPT2026_SE_KEY_MISMATCH = 5
};

struct FPT2026SealEmbeddedState {
    Parms parms;
    ZZ *mempool;
    SE_PTRS ptrs;
    SE_PRNG shareable_prng;
    SE_PRNG error_prng;
};

static const uint32_t fpt2026_moduli[FPT2026_SEAL_EMBEDDED_MODULI] = {
    1053818881u,
    1054015489u,
    1054212097u,
    1055260673u,
    1056178177u,
    1056440321u,
};

static int set_error(
    int status,
    char *message,
    size_t capacity,
    const char *text)
{
    if (message != NULL && capacity != 0) {
        (void)snprintf(message, capacity, "%s", text != NULL ? text : "unknown error");
    }
    return status;
}

static int validate_state(
    const FPT2026SealEmbeddedState *state,
    char *message,
    size_t capacity)
{
    if (state == NULL || state->mempool == NULL) {
        return set_error(FPT2026_SE_INVALID_ARGUMENT, message, capacity,
                         "SEAL-Embedded state is null");
    }
    if (state->parms.coeff_count != FPT2026_SEAL_EMBEDDED_DEGREE ||
        state->parms.nprimes != FPT2026_SEAL_EMBEDDED_MODULI) {
        return set_error(FPT2026_SE_SETUP_ERROR, message, capacity,
                         "SEAL-Embedded state has the wrong parameter shape");
    }
    return FPT2026_SE_OK;
}

static int load_compact_key(
    const char *path,
    ZZ *loaded_key,
    char *message,
    size_t capacity)
{
    const size_t key_bytes = FPT2026_SEAL_EMBEDDED_DEGREE / 4u;
    FILE *stream;
    int trailing;

    if (path == NULL || path[0] == '\0') {
        return set_error(FPT2026_SE_INVALID_ARGUMENT, message, capacity,
                         "compact key path is empty");
    }
    stream = fopen(path, "rb");
    if (stream == NULL) {
        return set_error(FPT2026_SE_SETUP_ERROR, message, capacity,
                         "cannot open compact benchmark key");
    }
    memset(loaded_key, 0, FPT2026_SEAL_EMBEDDED_DEGREE * sizeof(*loaded_key));
    if (fread((unsigned char *)loaded_key, 1u, key_bytes, stream) != key_bytes) {
        fclose(stream);
        return set_error(FPT2026_SE_SETUP_ERROR, message, capacity,
                         "compact benchmark key has the wrong size");
    }
    trailing = fgetc(stream);
    fclose(stream);
    if (trailing != EOF) {
        return set_error(FPT2026_SE_SETUP_ERROR, message, capacity,
                         "compact benchmark key has trailing bytes");
    }
    return FPT2026_SE_OK;
}

int fpt2026_seal_embedded_create(
    const FPT2026SealEmbeddedConfig *config,
    FPT2026SealEmbeddedState **state,
    char *error_message,
    size_t error_message_capacity)
{
    FPT2026SealEmbeddedState *created;
    uint8_t setup_seed[SE_PRNG_SEED_BYTE_COUNT] = {0};
    size_t p;
    int status;

    if (state != NULL) {
        *state = NULL;
    }
    if (config == NULL || state == NULL) {
        return set_error(FPT2026_SE_INVALID_ARGUMENT, error_message,
                         error_message_capacity, "config or output state is null");
    }
    if (config->abi_version != FPT2026_SEAL_EMBEDDED_C_ABI_VERSION ||
        config->struct_size < sizeof(*config)) {
        return set_error(FPT2026_SE_ABI_MISMATCH, error_message,
                         error_message_capacity, "SEAL-Embedded config ABI mismatch");
    }
    if (config->degree != FPT2026_SEAL_EMBEDDED_DEGREE ||
        config->num_moduli != FPT2026_SEAL_EMBEDDED_MODULI ||
        config->scale != 33554432.0) {
        return set_error(FPT2026_SE_INVALID_ARGUMENT, error_message,
                         error_message_capacity, "FPT 2026 requires N=8192, six moduli, scale 2^25");
    }

    created = (FPT2026SealEmbeddedState *)calloc(1u, sizeof(*created));
    if (created == NULL) {
        return set_error(FPT2026_SE_SETUP_ERROR, error_message,
                         error_message_capacity, "cannot allocate SEAL-Embedded state");
    }
    created->parms.is_asymmetric = false;
    created->parms.small_s = true;
    created->parms.sample_s = false;
    created->mempool = ckks_mempool_setup_sym(config->degree);
    if (created->mempool == NULL) {
        free(created);
        return set_error(FPT2026_SE_SETUP_ERROR, error_message,
                         error_message_capacity, "cannot allocate SEAL-Embedded memory pool");
    }
    ckks_set_ptrs_sym(config->degree, created->mempool, &created->ptrs);
    ckks_setup(config->degree, config->num_moduli,
               created->ptrs.index_map_ptr, &created->parms);

    for (p = 0; p < FPT2026_SEAL_EMBEDDED_MODULI; ++p) {
        if (created->parms.moduli[p].value != fpt2026_moduli[p]) {
            fpt2026_seal_embedded_destroy(created);
            return set_error(FPT2026_SE_SETUP_ERROR, error_message,
                             error_message_capacity, "SEAL-Embedded modulus contract mismatch");
        }
    }
    /* Exercise the stock public setup path without its hidden SE_DATA_PATH file
       dependency, then replace the sampled packed key with the explicit pinned
       compact key that controls every benchmark operation. */
    created->parms.sample_s = true;
    ckks_setup_s(&created->parms, setup_seed, &created->error_prng,
                 created->ptrs.ternary);
    created->parms.sample_s = false;
    status = load_compact_key(config->compact_key_path, created->ptrs.ternary,
                              error_message, error_message_capacity);
    if (status != FPT2026_SE_OK) {
        fpt2026_seal_embedded_destroy(created);
        return status;
    }
    *state = created;
    return FPT2026_SE_OK;
}

int fpt2026_seal_embedded_copy_parameters(
    const FPT2026SealEmbeddedState *state,
    uint32_t *moduli,
    size_t moduli_capacity,
    uint32_t *const_ratios,
    size_t ratio_capacity,
    char *error_message,
    size_t error_message_capacity)
{
    size_t p;
    int status = validate_state(state, error_message, error_message_capacity);
    if (status != FPT2026_SE_OK) {
        return status;
    }
    if (moduli == NULL || moduli_capacity < FPT2026_SEAL_EMBEDDED_MODULI ||
        const_ratios == NULL || ratio_capacity < 2u * FPT2026_SEAL_EMBEDDED_MODULI) {
        return set_error(FPT2026_SE_INVALID_ARGUMENT, error_message,
                         error_message_capacity, "parameter output capacity is insufficient");
    }
    for (p = 0; p < FPT2026_SEAL_EMBEDDED_MODULI; ++p) {
        moduli[p] = state->parms.moduli[p].value;
        const_ratios[2u * p] = state->parms.moduli[p].const_ratio[0];
        const_ratios[2u * p + 1u] = state->parms.moduli[p].const_ratio[1];
    }
    return FPT2026_SE_OK;
}

int fpt2026_seal_embedded_copy_index_map(
    const FPT2026SealEmbeddedState *state,
    uint16_t *index_map,
    size_t index_map_capacity,
    char *error_message,
    size_t error_message_capacity)
{
    int status = validate_state(state, error_message, error_message_capacity);
    if (status != FPT2026_SE_OK) {
        return status;
    }
    if (index_map == NULL || index_map_capacity < FPT2026_SEAL_EMBEDDED_DEGREE) {
        return set_error(FPT2026_SE_INVALID_ARGUMENT, error_message,
                         error_message_capacity, "index-map output capacity is insufficient");
    }
    memcpy(index_map, state->ptrs.index_map_ptr,
           FPT2026_SEAL_EMBEDDED_DEGREE * sizeof(*index_map));
    return FPT2026_SE_OK;
}

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
    size_t error_message_capacity)
{
    size_t p;
    uint8_t share_seed_copy[FPT2026_SEAL_EMBEDDED_SEED_BYTES];
    uint8_t error_seed_copy[FPT2026_SEAL_EMBEDDED_SEED_BYTES];
    const size_t output_count = FPT2026_SEAL_EMBEDDED_DEGREE *
                                FPT2026_SEAL_EMBEDDED_MODULI;
    int status = validate_state(state, error_message, error_message_capacity);
    if (status != FPT2026_SE_OK) {
        return status;
    }
    if (values == NULL || value_count != FPT2026_SEAL_EMBEDDED_DEGREE / 2u ||
        shareable_seed == NULL || error_seed == NULL || c0 == NULL || c1 == NULL ||
        c0_capacity < output_count || c1_capacity < output_count || result == NULL) {
        return set_error(FPT2026_SE_INVALID_ARGUMENT, error_message,
                         error_message_capacity, "invalid SEAL-Embedded encryption buffers");
    }
    if (result->abi_version != FPT2026_SEAL_EMBEDDED_C_ABI_VERSION ||
        result->struct_size < sizeof(*result)) {
        return set_error(FPT2026_SE_ABI_MISMATCH, error_message,
                         error_message_capacity, "SEAL-Embedded result ABI mismatch");
    }

    memcpy(share_seed_copy, shareable_seed, sizeof(share_seed_copy));
    memcpy(error_seed_copy, error_seed, sizeof(error_seed_copy));
    result->modulus_visit_count = 0;
    ckks_reset_primes(&state->parms);
    if (!ckks_encode_base(&state->parms, values, value_count,
                          state->ptrs.index_map_ptr, state->ptrs.ifft_roots,
                          state->ptrs.conj_vals)) {
        return set_error(FPT2026_SE_ENCODE_ERROR, error_message,
                         error_message_capacity, "SEAL-Embedded encoding failed");
    }
    ckks_sym_init(&state->parms, share_seed_copy, error_seed_copy,
                  &state->shareable_prng, &state->error_prng,
                  state->ptrs.conj_vals_int_ptr);

    for (p = 0; p < FPT2026_SEAL_EMBEDDED_MODULI; ++p) {
        result->modulus_visits[result->modulus_visit_count++] =
            (uint32_t)state->parms.curr_modulus_idx;
        ckks_encode_encrypt_sym(&state->parms, state->ptrs.conj_vals_int_ptr,
                                NULL, &state->shareable_prng, state->ptrs.ternary,
                                state->ptrs.ntt_pte_ptr, state->ptrs.ntt_roots_ptr,
                                state->ptrs.c0_ptr, state->ptrs.c1_ptr, NULL, NULL);
        memcpy(c0 + p * FPT2026_SEAL_EMBEDDED_DEGREE, state->ptrs.c0_ptr,
               FPT2026_SEAL_EMBEDDED_DEGREE * sizeof(ZZ));
        memcpy(c1 + p * FPT2026_SEAL_EMBEDDED_DEGREE, state->ptrs.c1_ptr,
               FPT2026_SEAL_EMBEDDED_DEGREE * sizeof(ZZ));
        if (p + 1u < FPT2026_SEAL_EMBEDDED_MODULI &&
            !ckks_next_prime_sym(&state->parms, state->ptrs.ternary)) {
            return set_error(FPT2026_SE_SETUP_ERROR, error_message,
                             error_message_capacity, "modulus chain ended before the sixth prime");
        }
    }
    return FPT2026_SE_OK;
}

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
    size_t error_message_capacity)
{
    size_t frame;
    size_t p;
    size_t frame_coefficients;
    size_t modulus_coefficients;
    int status = validate_state(state, error_message, error_message_capacity);
    if (status != FPT2026_SE_OK) {
        return status;
    }
    if (frame_count == 0 ||
        frame_count > SIZE_MAX / FPT2026_SEAL_EMBEDDED_DEGREE ||
        frame_count > SIZE_MAX / FPT2026_SEAL_EMBEDDED_SEED_BYTES) {
        return set_error(FPT2026_SE_INVALID_ARGUMENT, error_message,
                         error_message_capacity, "accelerator-preparation size overflow");
    }
    frame_coefficients = frame_count * FPT2026_SEAL_EMBEDDED_DEGREE;
    if (frame_coefficients > SIZE_MAX / FPT2026_SEAL_EMBEDDED_MODULI) {
        return set_error(FPT2026_SE_INVALID_ARGUMENT, error_message,
                         error_message_capacity, "accelerator-preparation modulus size overflow");
    }
    modulus_coefficients = FPT2026_SEAL_EMBEDDED_MODULI * frame_coefficients;
    if (shareable_seeds == NULL || error_seeds == NULL ||
        shareable_seed_bytes < frame_count * FPT2026_SEAL_EMBEDDED_SEED_BYTES ||
        error_seed_bytes < frame_count * FPT2026_SEAL_EMBEDDED_SEED_BYTES ||
        error_samples == NULL || error_capacity < frame_coefficients ||
        secret_keys == NULL || secret_key_capacity < modulus_coefficients ||
        uniform_polys == NULL || uniform_capacity < modulus_coefficients) {
        return set_error(FPT2026_SE_INVALID_ARGUMENT, error_message,
                         error_message_capacity, "invalid accelerator-preparation buffers");
    }

    for (frame = 0; frame < frame_count; ++frame) {
        uint8_t share_seed_copy[FPT2026_SEAL_EMBEDDED_SEED_BYTES];
        uint8_t error_seed_copy[FPT2026_SEAL_EMBEDDED_SEED_BYTES];
        memcpy(share_seed_copy,
               shareable_seeds + frame * FPT2026_SEAL_EMBEDDED_SEED_BYTES,
               sizeof(share_seed_copy));
        memcpy(error_seed_copy,
               error_seeds + frame * FPT2026_SEAL_EMBEDDED_SEED_BYTES,
               sizeof(error_seed_copy));
        prng_randomize_reset(&state->shareable_prng, share_seed_copy);
        prng_randomize_reset(&state->error_prng, error_seed_copy);
        sample_poly_cbd_generic_prng_16(
            FPT2026_SEAL_EMBEDDED_DEGREE, &state->error_prng,
            error_samples + frame * FPT2026_SEAL_EMBEDDED_DEGREE);

        ckks_reset_primes(&state->parms);
        for (p = 0; p < FPT2026_SEAL_EMBEDDED_MODULI; ++p) {
            const size_t offset = p * frame_coefficients +
                                  frame * FPT2026_SEAL_EMBEDDED_DEGREE;
            expand_poly_ternary(state->ptrs.ternary, &state->parms,
                                secret_keys + offset);
            sample_poly_uniform(&state->parms, &state->shareable_prng,
                                uniform_polys + offset);
            if (p + 1u < FPT2026_SEAL_EMBEDDED_MODULI &&
                !ckks_next_prime_sym(&state->parms, NULL)) {
                return set_error(FPT2026_SE_SETUP_ERROR, error_message,
                                 error_message_capacity,
                                 "modulus chain ended during accelerator preparation");
            }
        }
    }
    return FPT2026_SE_OK;
}

void fpt2026_seal_embedded_destroy(FPT2026SealEmbeddedState *state)
{
    if (state == NULL) {
        return;
    }
    if (state->parms.moduli != NULL) {
        delete_parameters(&state->parms);
    }
    free(state->mempool);
    state->mempool = NULL;
    free(state);
}
