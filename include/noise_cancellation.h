/**
 * noise_cancellation.h — C ABI for the noise_cancellation library
 *
 * Dual engine:
 *   - RNNoise (default): fully offline, weights embedded
 *   - DeepFilterNet3: higher quality, uses local models/ ONNX files
 *
 * Sample format: mono float32.
 * Host sample rate may be any positive rate; engines run at 48 kHz with
 * transparent streaming resampling.
 * Preferred block size at 48 kHz: 480 samples (10 ms).
 *
 * Threading: one NcHandle per audio stream.
 *
 * Build:
 *   cargo build --release --features "rnnoise,c-api"
 *   cargo build --release --features "rnnoise,deepfilternet,c-api"
 */

#ifndef NOISE_CANCELLATION_H
#define NOISE_CANCELLATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define NC_VERSION_MAJOR 0
#define NC_VERSION_MINOR 6
#define NC_VERSION_PATCH 0

/** Engine internal sample rate (Hz). */
#define NC_SAMPLE_RATE 48000

/** Preferred frame size at 48 kHz (samples). */
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
    NC_ENGINE_RNNOISE = 0,
    NC_ENGINE_DEEPFILTERNET = 1
} NcEngine;

typedef struct NcHandle NcHandle;

/** Create at 48 kHz host rate. */
NcStatus nc_create(NcEngine engine,
                   const char *model_dir,
                   float strength,
                   NcHandle **out);

/**
 * Create with host sample rate (Hz). Audio is resampled to 48 kHz for the
 * engine and back. Common rates: 8000, 16000, 22050, 32000, 44100, 48000.
 */
NcStatus nc_create_with_rate(NcEngine engine,
                             const char *model_dir,
                             float strength,
                             uint32_t sample_rate,
                             NcHandle **out);

void nc_destroy(NcHandle *handle);

/** Process mono f32 at the handle's host sample rate. In-place OK. */
NcStatus nc_process(NcHandle *handle,
                    const float *input,
                    float *output,
                    size_t num_samples);

NcStatus nc_reset(NcHandle *handle);
NcStatus nc_set_strength(NcHandle *handle, float strength);
NcStatus nc_get_strength(const NcHandle *handle, float *out_strength);
NcStatus nc_latency_samples(const NcHandle *handle, size_t *out_latency);
NcStatus nc_engine_name(const NcHandle *handle, const char **out_name);

uint32_t nc_sample_rate(void);
size_t nc_frame_size(void);
const char *nc_status_string(NcStatus status);

#ifdef __cplusplus
}
#endif

#endif /* NOISE_CANCELLATION_H */
