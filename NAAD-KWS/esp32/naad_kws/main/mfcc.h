// mfcc.h — on-device MFCC feature extraction.
//
// IMPORTANT: these constants must match training/config.py exactly, or the
// int8 model (trained on PC-computed MFCCs) will see out-of-distribution
// features on-device and accuracy will collapse. If you change config.py,
// mirror the change here.
#ifndef NAAD_MFCC_H_
#define NAAD_MFCC_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MFCC_SAMPLE_RATE          16000
#define MFCC_WINDOW_SIZE_SAMPLES  480    // 30 ms @ 16kHz
#define MFCC_WINDOW_STRIDE_SAMPLES 320   // 20 ms @ 16kHz  (hop)
#define MFCC_FFT_LEN              512    // next pow2 >= 480
#define MFCC_NUM_MEL_BINS         26
#define MFCC_NUM_COEFFS           13     // matches config.NUM_MFCC
#define MFCC_NUM_FRAMES           49     // matches config.NUM_FRAMES (1s clip)

// Must match the mean/std printed by train.py ("Feature normalization: ...")
// and saved to training/artifacts/feature_{mean,std}.npy.
// TODO: paste the real values here after training — convert.py prints a
// reminder, and you can also read them straight from the .npy files.
#define MFCC_FEATURE_MEAN   0.0f
#define MFCC_FEATURE_STD    1.0f

// One-time setup: precomputes the Hamming window, mel filterbank and DCT-II
// matrices. Call once at boot before any mfcc_compute_frame() calls.
void mfcc_init(void);

// Computes MFCC_NUM_COEFFS coefficients for one MFCC_WINDOW_SIZE_SAMPLES
// window of int16 PCM audio (values in native I2S range, NOT pre-scaled).
// out_coeffs must have room for MFCC_NUM_COEFFS floats.
void mfcc_compute_frame(const int16_t *pcm_window, float *out_coeffs);

#ifdef __cplusplus
}
#endif

#endif  // NAAD_MFCC_H_
