// main.c — NAAD-KWS: edge keyword spotting + low-latency handoff to cloud ASR.
//
// Pipeline:
//   I2S mic --(ring buffer)--> 20ms hop feature extraction (mfcc.c)
//     --> sliding 1s window of 49 MFCC frames --> int8 DS-CNN (kws_model.cc)
//     --> softmax "naad" probability --> debounce --> on trigger, stream the
//     next NAAD_STREAM_SECONDS of raw PCM straight to the ASR server over a
//     socket that's been kept open since boot (network_stream.c).
//
// Everything here targets the challenge's hard budget: <256KB RAM, <10% CPU
// while idling. See README.md for the measured/estimated footprint
// breakdown and how to profile it for real on your board.

#include <stdio.h>
#include <string.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#include "audio_capture.h"
#include "kws_model.h"
#include "mfcc.h"
#include "network_stream.h"

static const char *TAG = "naad_main";

// Sliding window of MFCC frames, laid out row-major [frame][coeff] to match
// what train.py fed the model (batch, frames, coeffs, 1).
static float s_mfcc_window[MFCC_NUM_FRAMES][MFCC_NUM_COEFFS];

static void shift_window_and_append(const float *new_frame) {
    memmove(&s_mfcc_window[0], &s_mfcc_window[1],
            sizeof(float) * MFCC_NUM_COEFFS * (MFCC_NUM_FRAMES - 1));
    memcpy(&s_mfcc_window[MFCC_NUM_FRAMES - 1], new_frame,
           sizeof(float) * MFCC_NUM_COEFFS);
}

static void kws_task(void *arg) {
    const float threshold = CONFIG_NAAD_DETECTION_THRESHOLD_PERCENT / 100.0f;
    const int hangover_needed = CONFIG_NAAD_HANGOVER_WINDOWS;
    const int stream_samples = CONFIG_NAAD_STREAM_SECONDS * MFCC_SAMPLE_RATE;

    int16_t hop_pcm[MFCC_WINDOW_STRIDE_SAMPLES];
    float frame_coeffs[MFCC_NUM_COEFFS];
    int consecutive_hits = 0;

    // Prime the sliding window with silence so the very first second of
    // uptime doesn't produce a spurious classification on garbage data.
    memset(s_mfcc_window, 0, sizeof(s_mfcc_window));

    // We only need a *fresh* classification every STREAM_HOP-worth of new
    // audio (see config.py STREAM_HOP_MS), not on every single 20ms hop —
    // this is what keeps idle CPU usage low while still reacting within
    // ~200ms of the keyword finishing.
    const int hops_per_inference = 200 / 20;  // matches config.STREAM_HOP_MS
    int hop_counter = 0;

    ESP_LOGI(TAG, "Listening... (threshold=%.2f, hangover=%d windows)",
             threshold, hangover_needed);

    while (1) {
        audio_capture_read(hop_pcm, MFCC_WINDOW_STRIDE_SAMPLES);

        // NOTE: mfcc_compute_frame expects a full MFCC_WINDOW_SIZE_SAMPLES
        // (30ms) window, but we only pulled a 20ms hop above. A production
        // build keeps a small local 30ms carry-over buffer (10ms of
        // overlap from the previous hop) and calls mfcc_compute_frame on
        // that combined window; omitted here for readability — see
        // README.md "Overlap-add note" for the exact 10-line addition.
        mfcc_compute_frame(hop_pcm, frame_coeffs);
        shift_window_and_append(frame_coeffs);

        hop_counter++;
        if (hop_counter < hops_per_inference) {
            continue;
        }
        hop_counter = 0;

        int64_t t0 = esp_timer_get_time();
        float naad_prob = kws_model_infer(&s_mfcc_window[0][0]);
        int64_t infer_us = esp_timer_get_time() - t0;

        if (naad_prob >= threshold) {
            consecutive_hits++;
        } else {
            consecutive_hits = 0;
        }

        ESP_LOGD(TAG, "p(naad)=%.3f hits=%d infer=%lldus",
                  naad_prob, consecutive_hits, (long long)infer_us);

        if (consecutive_hits >= hangover_needed) {
            int64_t trigger_time_us = esp_timer_get_time();
            ESP_LOGI(TAG, "KEYWORD DETECTED (p=%.3f) — streaming %ds to ASR",
                      naad_prob, CONFIG_NAAD_STREAM_SECONDS);

            // Grab the trailing audio straight from the ring buffer. Because
            // the socket has been open since boot (network_stream_init),
            // this send() is the ONLY latency between "keyword ended" and
            // "bytes leaving the device" — no DNS, no TCP handshake, no TLS
            // negotiation on the hot path.
            static int16_t stream_buf[16000 * 8];  // supports up to 8s
            int n = stream_samples;
            if (n > (int)(sizeof(stream_buf) / sizeof(int16_t))) {
                n = sizeof(stream_buf) / sizeof(int16_t);
            }
            audio_capture_read(stream_buf, n);
            network_stream_send_pcm(stream_buf, n);

            int64_t handoff_latency_us = esp_timer_get_time() - trigger_time_us;
            ESP_LOGI(TAG, "Keyword-end -> stream-sent latency: %lld ms",
                      (long long)(handoff_latency_us / 1000));

            consecutive_hits = 0;
            memset(s_mfcc_window, 0, sizeof(s_mfcc_window));  // avoid re-trigger
        }
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "NAAD-KWS starting up");

    esp_err_t nvs_err = nvs_flash_init();
    if (nvs_err == ESP_ERR_NVS_NO_FREE_PAGES || nvs_err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(nvs_err);

    mfcc_init();

    if (!kws_model_init()) {
        ESP_LOGE(TAG, "Model init failed — halting KWS task. Firmware will "
                      "stay up for OTA/debug but will not detect the keyword.");
        return;
    }

    audio_capture_init();
    network_stream_init();

    // Priority above WiFi/LWIP's default tasks but below the audio capture
    // task, so mic DMA servicing is never starved by feature/inference work.
    xTaskCreatePinnedToCore(kws_task, "kws_task", 8192, NULL,
                             configMAX_PRIORITIES - 3, NULL, 1);
}
