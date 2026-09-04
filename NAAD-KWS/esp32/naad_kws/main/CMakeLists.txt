idf_component_register(
    SRCS
        "main.c"
        "audio_capture.c"
        "mfcc.c"
        "network_stream.c"
        "model_data.cc"
        "kws_model.cc"
    INCLUDE_DIRS "."
    REQUIRES
        driver          # I2S
        esp_wifi
        esp_netif
        nvs_flash
        esp_event
        lwip
)

# TFLite-Micro headers are C++; main.c stays C and calls into a thin
# C++ wrapper (kws_model.cc) — see kws_model.h.
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=maybe-uninitialized)
