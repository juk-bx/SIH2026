// mfcc.c — on-device MFCC feature extraction using esp-dsp's real FFT.
//
// Pipeline per 30ms window (matches librosa.feature.mfcc on the Python
// side, see training/audio_utils.py):
//   1. Hamming window
//   2. Zero-pad to MFCC_FFT_LEN, real FFT (esp-dsp)
//   3. Power spectrum -> mel filterbank (triangular filters) -> log
//   4. DCT-II -> take first MFCC_NUM_COEFFS coefficients
//
// All filterbank / DCT matrices are computed once in mfcc_init() rather
// than baked in as giant flash tables — it's a few hundred microseconds
// at boot and keeps this file self-contained.

#include "mfcc.h"

#include <math.h>
#include <string.h>

#include "esp_dsp.h"
#include "esp_log.h"

static const char *TAG = "mfcc";

#define M_PI_F 3.14159265358979323846f

static float s_hamming[MFCC_WINDOW_SIZE_SAMPLES];
static float s_mel_filterbank[MFCC_NUM_MEL_BINS][MFCC_FFT_LEN / 2 + 1];
static float s_dct_matrix[MFCC_NUM_COEFFS][MFCC_NUM_MEL_BINS];
static bool s_initialized = false;

// scratch buffers (module-static so they aren't re-allocated per call —
// keeps this hot path allocation-free, important for the "idle listening"
// CPU/latency budget)
static float s_fft_buf[MFCC_FFT_LEN * 2];   // interleaved re/im for esp-dsp
static float s_mel_energy[MFCC_NUM_MEL_BINS];

static inline float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void build_mel_filterbank(void) {
    const int n_fft_bins = MFCC_FFT_LEN / 2 + 1;
    const float mel_low = hz_to_mel(0.0f);
    const float mel_high = hz_to_mel(MFCC_SAMPLE_RATE / 2.0f);

    float mel_points[MFCC_NUM_MEL_BINS + 2];
    for (int i = 0; i < MFCC_NUM_MEL_BINS + 2; i++) {
        mel_points[i] = mel_low + (mel_high - mel_low) * i / (MFCC_NUM_MEL_BINS + 1);
    }

    int bin[MFCC_NUM_MEL_BINS + 2];
    for (int i = 0; i < MFCC_NUM_MEL_BINS + 2; i++) {
        float hz = mel_to_hz(mel_points[i]);
        bin[i] = (int)floorf((MFCC_FFT_LEN + 1) * hz / MFCC_SAMPLE_RATE);
    }

    memset(s_mel_filterbank, 0, sizeof(s_mel_filterbank));
    for (int m = 1; m <= MFCC_NUM_MEL_BINS; m++) {
        int f_left = bin[m - 1];
        int f_center = bin[m];
        int f_right = bin[m + 1];

        for (int k = f_left; k < f_center && k < n_fft_bins; k++) {
            if (k < 0) continue;
            s_mel_filterbank[m - 1][k] =
                (float)(k - f_left) / (float)(f_center - f_left + 1e-9f);
        }
        for (int k = f_center; k < f_right && k < n_fft_bins; k++) {
            if (k < 0) continue;
            s_mel_filterbank[m - 1][k] =
                (float)(f_right - k) / (float)(f_right - f_center + 1e-9f);
        }
    }
}

static void build_dct_matrix(void) {
    // Orthonormal DCT-II, matching librosa's default (norm="ortho").
    for (int k = 0; k < MFCC_NUM_COEFFS; k++) {
        for (int n = 0; n < MFCC_NUM_MEL_BINS; n++) {
            float val = cosf(M_PI_F / MFCC_NUM_MEL_BINS * (n + 0.5f) * k);
            float scale = (k == 0)
                ? sqrtf(1.0f / MFCC_NUM_MEL_BINS)
                : sqrtf(2.0f / MFCC_NUM_MEL_BINS);
            s_dct_matrix[k][n] = val * scale;
        }
    }
}

void mfcc_init(void) {
    // Hamming window
    for (int i = 0; i < MFCC_WINDOW_SIZE_SAMPLES; i++) {
        s_hamming[i] = 0.54f - 0.46f * cosf(2.0f * M_PI_F * i /
                                             (MFCC_WINDOW_SIZE_SAMPLES - 1));
    }

    build_mel_filterbank();
    build_dct_matrix();

    esp_err_t err = dsps_fft2r_init_fc32(NULL, MFCC_FFT_LEN);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "dsps_fft2r_init_fc32 failed: %d", err);
    }

    s_initialized = true;
    ESP_LOGI(TAG, "mfcc_init done (fft_len=%d, mel_bins=%d, coeffs=%d)",
             MFCC_FFT_LEN, MFCC_NUM_MEL_BINS, MFCC_NUM_COEFFS);
}

void mfcc_compute_frame(const int16_t *pcm_window, float *out_coeffs) {
    if (!s_initialized) {
        ESP_LOGE(TAG, "mfcc_compute_frame called before mfcc_init()");
        return;
    }

    // 1. Window + zero-pad, pack as interleaved complex (im=0) for esp-dsp.
    memset(s_fft_buf, 0, sizeof(s_fft_buf));
    for (int i = 0; i < MFCC_WINDOW_SIZE_SAMPLES; i++) {
        float sample = (float)pcm_window[i] / 32768.0f;   // int16 -> [-1, 1)
        s_fft_buf[2 * i] = sample * s_hamming[i];
        s_fft_buf[2 * i + 1] = 0.0f;
    }

    // 2. In-place radix-2 FFT (esp-dsp, uses ESP32 SIMD instructions where
    //    available — this is the single most expensive op per frame).
    dsps_fft2r_fc32(s_fft_buf, MFCC_FFT_LEN);
    dsps_bit_rev_fc32(s_fft_buf, MFCC_FFT_LEN);

    // 3. Power spectrum for the first half (real input -> symmetric spectrum)
    const int n_fft_bins = MFCC_FFT_LEN / 2 + 1;
    static float power_spec[MFCC_FFT_LEN / 2 + 1];
    for (int k = 0; k < n_fft_bins; k++) {
        float re = s_fft_buf[2 * k];
        float im = s_fft_buf[2 * k + 1];
        power_spec[k] = (re * re + im * im) / MFCC_FFT_LEN;
    }

    // 4. Mel filterbank + log
    for (int m = 0; m < MFCC_NUM_MEL_BINS; m++) {
        float energy = 0.0f;
        for (int k = 0; k < n_fft_bins; k++) {
            energy += s_mel_filterbank[m][k] * power_spec[k];
        }
        s_mel_energy[m] = logf(energy + 1e-6f);
    }

    // 5. DCT-II -> MFCC coefficients
    for (int c = 0; c < MFCC_NUM_COEFFS; c++) {
        float sum = 0.0f;
        for (int m = 0; m < MFCC_NUM_MEL_BINS; m++) {
            sum += s_dct_matrix[c][m] * s_mel_energy[m];
        }
        out_coeffs[c] = sum;
    }
}
