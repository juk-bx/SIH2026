// network_stream.c — see network_stream.h for design notes.

#include "network_stream.h"

#include <errno.h>
#include <stdbool.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/netdb.h"
#include "sdkconfig.h"

static const char *TAG = "network_stream";

#define WIFI_CONNECTED_BIT BIT0
static EventGroupHandle_t s_wifi_event_group;

static int s_sock = -1;
static SemaphoreHandle_t s_sock_mutex;

// Trivial framing so the ASR-side reference server can tell where one
// utterance ends and the next begins on a long-lived TCP stream, without
// adding meaningful overhead (8 bytes per utterance, not per packet).
typedef struct __attribute__((packed)) {
    uint32_t magic;         // 'NAAD' = 0x4441414E (little-endian on wire)
    uint32_t num_samples;   // int16 samples following this header
} naad_frame_header_t;

#define NAAD_FRAME_MAGIC 0x4441414Eu

static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "WiFi disconnected, retrying...");
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ESP_LOGI(TAG, "WiFi connected, got IP");
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static bool connect_socket(void) {
    struct sockaddr_in dest = {
        .sin_family = AF_INET,
        .sin_port = htons(CONFIG_NAAD_ASR_SERVER_PORT),
    };
    dest.sin_addr.s_addr = inet_addr(CONFIG_NAAD_ASR_SERVER_IP);

    int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0) {
        ESP_LOGE(TAG, "socket() failed: errno %d", errno);
        return false;
    }

    // Disable Nagle's algorithm — we want each utterance flushed to the
    // network immediately, not batched, to keep keyword->ASR latency low.
    int one = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    if (connect(sock, (struct sockaddr *)&dest, sizeof(dest)) != 0) {
        ESP_LOGW(TAG, "connect() to ASR server failed: errno %d", errno);
        close(sock);
        return false;
    }

    xSemaphoreTake(s_sock_mutex, portMAX_DELAY);
    s_sock = sock;
    xSemaphoreGive(s_sock_mutex);

    ESP_LOGI(TAG, "Connected to ASR server %s:%d",
             CONFIG_NAAD_ASR_SERVER_IP, CONFIG_NAAD_ASR_SERVER_PORT);
    return true;
}

static void connection_manager_task(void *arg) {
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE,
                         portMAX_DELAY);

    int backoff_ms = 500;
    while (1) {
        xSemaphoreTake(s_sock_mutex, portMAX_DELAY);
        bool need_connect = (s_sock < 0);
        xSemaphoreGive(s_sock_mutex);

        if (need_connect) {
            if (connect_socket()) {
                backoff_ms = 500;
            } else {
                vTaskDelay(pdMS_TO_TICKS(backoff_ms));
                backoff_ms = backoff_ms < 8000 ? backoff_ms * 2 : 8000;
            }
        } else {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }
}

void network_stream_init(void) {
    s_sock_mutex = xSemaphoreCreateMutex();
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                                &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                                &wifi_event_handler, NULL));

    wifi_config_t wifi_config = {0};
    strncpy((char *)wifi_config.sta.ssid, CONFIG_NAAD_WIFI_SSID,
            sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char *)wifi_config.sta.password, CONFIG_NAAD_WIFI_PASSWORD,
            sizeof(wifi_config.sta.password) - 1);

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    xTaskCreate(connection_manager_task, "net_conn_mgr", 4096, NULL, 5, NULL);
}

void network_stream_send_pcm(const int16_t *pcm, size_t num_samples) {
    xSemaphoreTake(s_sock_mutex, portMAX_DELAY);
    int sock = s_sock;
    xSemaphoreGive(s_sock_mutex);

    if (sock < 0) {
        ESP_LOGW(TAG, "no ASR connection available, dropping %d samples",
                 (int)num_samples);
        return;
    }

    naad_frame_header_t hdr = {
        .magic = NAAD_FRAME_MAGIC,
        .num_samples = (uint32_t)num_samples,
    };

    int sent = send(sock, &hdr, sizeof(hdr), 0);
    if (sent != sizeof(hdr)) goto conn_error;

    const uint8_t *data = (const uint8_t *)pcm;
    size_t total_bytes = num_samples * sizeof(int16_t);
    size_t sent_bytes = 0;
    while (sent_bytes < total_bytes) {
        int n = send(sock, data + sent_bytes, total_bytes - sent_bytes, 0);
        if (n <= 0) goto conn_error;
        sent_bytes += (size_t)n;
    }

    ESP_LOGI(TAG, "Streamed %d samples (%.2fs) to ASR server",
              (int)num_samples, (float)num_samples / 16000.0f);
    return;

conn_error:
    ESP_LOGW(TAG, "send() failed (errno %d), will reconnect", errno);
    xSemaphoreTake(s_sock_mutex, portMAX_DELAY);
    close(s_sock);
    s_sock = -1;
    xSemaphoreGive(s_sock_mutex);
}

bool network_stream_is_connected(void) {
    xSemaphoreTake(s_sock_mutex, portMAX_DELAY);
    bool connected = (s_sock >= 0);
    xSemaphoreGive(s_sock_mutex);
    return connected;
}
