/**
 * noise_cancellation.h — C ABI for the noise_cancellation library
 *
 * Dual engine:
 *   - RNNoise (default): fully offline, weights embedded
 *   - DeepFilterNet3: higher quality, uses local models/ ONNX files
 *
 * Sample format: mono float32, 48 kHz.
 * Preferred block size: 480 samples (10 ms). Any length is accepted.
 *
 * Threading: one NcHandle per audio stream. Do not call process from
 * multiple threads on the same handle without external synchronisation.
 *
 * Build (Rust side):
 *   cargo build --release --features "rnnoise,c-api"
 *   cargo build --release --features "rnnoise,deepfilternet,c-api"
 *
 * Link against:
 *   target/release/libnoise_cancellation.a   (static)
 *   or libnoise_cancellation.so / .dylib     (shared, cdylib)
 */

#ifndef NOISE_CANCELLATION_H
#define NOISE_CANCELLATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define NC_VERSION_MAJOR 0
#define NC_VERSION_MINOR 5
#define NC_VERSION_PATCH 0

/** Library sample rate (Hz). All engines require this rate. */
#define NC_SAMPLE_RATE 48000

/** Preferred frame size in samples (10 ms @ 48 kHz). */
#define NC_FRAME_SIZE 480

typedef enum NcStatus {
    NC_OK = 0,
    NC_ERR_NULL_POINTER = -1,
    NC_ERR_INVALID_ARG = -2,
    NC_ERR_ENGINE = -3,
    NC_ERR_FEATURE_DISABLED = -4,
    NC_ERR_OUT_OF_MEMORY = -5,
    NC_ERR_UNSUPPORTED = -6
} NcStatus;

typedef enum NcEngine {
    /** Ultra-light RNNoise (default, no model files). */
    NC_ENGINE_RNNOISE = 0,
    /**
     * DeepFilterNet3. model_dir must point to a directory containing
     * enc.onnx, erb_dec.onnx, df_dec.onnx (or NULL to auto-discover).
     */
    NC_ENGINE_DEEPFILTERNET = 1
} NcEngine;

typedef struct NcHandle NcHandle;

/**
 * Create a denoiser instance.
 *
 * @param engine     NC_ENGINE_RNNOISE or NC_ENGINE_DEEPFILTERNET
 * @param model_dir  Path to DFN3 models directory, or NULL for auto-discover
 *                   / RNNoise. UTF-8. Ignored for RNNoise.
 * @param strength   Suppression strength in [0.0, 1.0]. Use 0.85 for default.
 * @param out        Receives the new handle on success. Must not be NULL.
 * @return           NC_OK or an error code.
 */
NcStatus nc_create(NcEngine engine,
                   const char *model_dir,
                   float strength,
                   NcHandle **out);

/** Destroy a handle and free all resources. Safe to call with NULL. */
void nc_destroy(NcHandle *handle);

/**
 * Process mono f32 samples @ 48 kHz.
 * input and output may be the same pointer (in-place).
 */
NcStatus nc_process(NcHandle *handle,
                    const float *input,
                    float *output,
                    size_t num_samples);

/** Reset internal state (e.g. when switching to an unrelated stream). */
NcStatus nc_reset(NcHandle *handle);

/** Set suppression strength in [0.0, 1.0]. */
NcStatus nc_set_strength(NcHandle *handle, float strength);

/** Get current strength. */
NcStatus nc_get_strength(const NcHandle *handle, float *out_strength);

/** Algorithmic latency in samples for the current engine. */
NcStatus nc_latency_samples(const NcHandle *handle, size_t *out_latency);

/**
 * Engine name string ("RNNoise" or "DeepFilterNet3").
 * Pointer is valid for the lifetime of the handle. Do not free.
 */
NcStatus nc_engine_name(const NcHandle *handle, const char **out_name);

/** Sample rate required by all engines (always 48000). */
uint32_t nc_sample_rate(void);

/** Preferred frame size (always 480). */
size_t nc_frame_size(void);

/** Human-readable status string. Static storage; do not free. */
const char *nc_status_string(NcStatus status);

#ifdef __cplusplus
}
#endif

#endif /* NOISE_CANCELLATION_H */
