/**
 * noise_cancellation.h — C ABI for the noise_cancellation library
 *
 * Dual engine:
 *   - RNNoise (default): fully offline, weights embedded
 *   - DeepFilterNet3: higher quality, uses local models/ ONNX files
 *
 * Sample format: mono or interleaved multi-channel float32.
 * Host sample rate may be any positive rate; engines run at 48 kHz with
 * transparent streaming resampling.
 *
 * Threading: one handle per audio stream.
 *
 * Build:
 *   cargo build --release --features "rnnoise,c-api"
 */

#ifndef NOISE_CANCELLATION_H
#define NOISE_CANCELLATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define NC_VERSION_MAJOR 0
#define NC_VERSION_MINOR 7
#define NC_VERSION_PATCH 0

#define NC_SAMPLE_RATE 48000
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

NcStatus nc_create(NcEngine engine,
                   const char *model_dir,
                   float strength,
                   NcHandle **out);

NcStatus nc_create_with_rate(NcEngine engine,
                             const char *model_dir,
                             float strength,
                             uint32_t sample_rate,
                             NcHandle **out);

void nc_destroy(NcHandle *handle);

NcStatus nc_process(NcHandle *handle,
                    const float *input,
                    float *output,
                    size_t num_samples);

/* Drain the delayed tail after the final process call. Repeat until out_written == 0. */
NcStatus nc_flush(NcHandle *handle,
                  float *output,
                  size_t capacity,
                  size_t *out_written);

NcStatus nc_reset(NcHandle *handle);
NcStatus nc_set_strength(NcHandle *handle, float strength);
NcStatus nc_get_strength(const NcHandle *handle, float *out_strength);
NcStatus nc_latency_samples(const NcHandle *handle, size_t *out_latency);
NcStatus nc_engine_name(const NcHandle *handle, const char **out_name);

uint32_t nc_sample_rate(void);
size_t nc_frame_size(void);
const char *nc_status_string(NcStatus status);

/* ---- Multi-channel (interleaved) ---- */

typedef struct NcMultiHandle NcMultiHandle;

NcStatus nc_create_multi(NcEngine engine,
                         const char *model_dir,
                         float strength,
                         uint32_t channels,
                         NcMultiHandle **out);

NcStatus nc_create_multi_with_rate(NcEngine engine,
                                   const char *model_dir,
                                   float strength,
                                   uint32_t sample_rate,
                                   uint32_t channels,
                                   NcMultiHandle **out);

void nc_destroy_multi(NcMultiHandle *handle);

NcStatus nc_process_interleaved(NcMultiHandle *handle,
                                const float *input,
                                float *output,
                                size_t frames);

/* Drain delayed interleaved tail frames. Repeat until out_frames == 0. */
NcStatus nc_flush_interleaved(NcMultiHandle *handle,
                              float *output,
                              size_t capacity_frames,
                              size_t *out_frames);

NcStatus nc_multi_channels(const NcMultiHandle *handle, uint32_t *out);
NcStatus nc_multi_reset(NcMultiHandle *handle);
NcStatus nc_multi_set_strength(NcMultiHandle *handle, float strength);

#ifdef __cplusplus
}
#endif

#endif /* NOISE_CANCELLATION_H */
