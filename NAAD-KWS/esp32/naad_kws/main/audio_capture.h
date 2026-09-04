// audio_capture.h — I2S microphone capture into a ring buffer that both the
// feature-extraction task and the "stream raw audio to ASR" path can read
// from independently.
//
// Wiring assumed: a digital I2S MEMS mic (e.g. INMP441 / ICS-43434), mono,
// 16-bit, 16kHz. Adjust the GPIO pins below for your board.
#ifndef NAAD_AUDIO_CAPTURE_H_
#define NAAD_AUDIO_CAPTURE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- Pin mapping — EDIT FOR YOUR BOARD ------------------------------------
#define NAAD_I2S_WS_GPIO    25   // LRCLK / word select
#define NAAD_I2S_SCK_GPIO   26   // BCLK / bit clock
#define NAAD_I2S_SD_GPIO    22   // DOUT from mic -> ESP32 input

// Ring buffer sized for ~1.5s of headroom at 16kHz/16-bit mono so the
// network task can drain a full utterance even if it's briefly scheduled
// late; keep this well under the 256KB RAM budget alongside the tensor
// arena and TCP buffers (see README.md for the full memory budget table).
#define NAAD_AUDIO_RINGBUF_BYTES (16000 * 2 * 3 / 2)  // ~48 KB

// Starts the I2S peripheral and the background task that continuously
// pumps DMA'd samples into the ring buffer. Call once at boot.
void audio_capture_init(void);

// Blocking read of exactly `num_samples` int16 samples (mono) into `out`.
// Returns the number of samples actually copied (< num_samples only if the
// capture task was torn down, which shouldn't happen in normal operation).
size_t audio_capture_read(int16_t *out, size_t num_samples);

// Non-blocking peek at how many samples are currently buffered.
size_t audio_capture_available(void);

#ifdef __cplusplus
}
#endif

#endif  // NAAD_AUDIO_CAPTURE_H_
