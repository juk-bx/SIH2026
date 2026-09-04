// audio_capture.c — see audio_capture.h for design notes.

#include "audio_capture.h"

#include <string.h>

#include "driver/i2s_std.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/ringbuf.h"
#include "freertos/task.h"

static const char *TAG = "audio_capture";

static i2s_chan_handle_t s_rx_chan = NULL;
static RingbufHandle_t s_ringbuf = NULL;

// Read this many samples from I2S per DMA pull; small enough to keep
// capture latency low, large enough to be efficient.
#define I2S_READ_CHUNK_SAMPLES 320   // 20 ms @ 16kHz — matches MFCC hop

static void audio_capture_task(void *arg) {
    int16_t chunk[I2S_READ_CHUNK_SAMPLES];
    size_t bytes_read = 0;

    while (1) {
        esp_err_t err = i2s_channel_read(
            s_rx_chan, chunk, sizeof(chunk), &bytes_read, portMAX_DELAY);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "i2s_channel_read failed: %d", err);
            continue;
        }

        if (xRingbufferSend(s_ringbuf, chunk, bytes_read, pdMS_TO_TICKS(50)) != pdTRUE) {
            // Ring buffer full — a downstream reader is falling behind.
            // Drop this chunk rather than blocking, so mic capture timing
            // (and therefore MFCC frame alignment) stays real-time.
            ESP_LOGW(TAG, "ring buffer full, dropping %d bytes", (int)bytes_read);
        }
    }
}

void audio_capture_init(void) {
    s_ringbuf = xRingbufferCreate(NAAD_AUDIO_RINGBUF_BYTES, RINGBUF_TYPE_BYTEBUF);
    if (!s_ringbuf) {
        ESP_LOGE(TAG, "failed to allocate ring buffer");
        return;
    }

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &s_rx_chan));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = NAAD_I2S_SCK_GPIO,
            .ws = NAAD_I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = NAAD_I2S_SD_GPIO,
            .invert_flags = {
                .mclk_inv = false, .bclk_inv = false, .ws_inv = false,
            },
        },
    };
    // Most I2S MEMS mics output on the LEFT channel only; if your mic is
    // wired to select RIGHT (L/R pin high), flip this.
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx_chan));

    // Pinned to core 0 with a modest stack; this task just pumps DMA -> ring
    // buffer, no floating point work, so it's cheap and keeps idle CPU low.
    xTaskCreatePinnedToCore(audio_capture_task, "audio_capture", 3072, NULL,
                             configMAX_PRIORITIES - 2, NULL, 0);

    ESP_LOGI(TAG, "audio capture started (I2S std mode, 16kHz mono 16-bit)");
}

size_t audio_capture_read(int16_t *out, size_t num_samples) {
    size_t bytes_needed = num_samples * sizeof(int16_t);
    size_t bytes_copied = 0;

    while (bytes_copied < bytes_needed) {
        size_t chunk_size = 0;
        void *item = xRingbufferReceiveUpTo(
            s_ringbuf, &chunk_size, portMAX_DELAY, bytes_needed - bytes_copied);
        if (!item) continue;

        memcpy((uint8_t *)out + bytes_copied, item, chunk_size);
        bytes_copied += chunk_size;
        vRingbufferReturnItem(s_ringbuf, item);
    }
    return bytes_copied / sizeof(int16_t);
}

size_t audio_capture_available(void) {
    UBaseType_t items_waiting = 0;
    vRingbufferGetInfo(s_ringbuf, NULL, NULL, NULL, NULL, &items_waiting);
    return items_waiting;
}
