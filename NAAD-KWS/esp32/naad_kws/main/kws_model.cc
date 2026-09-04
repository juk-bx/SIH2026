// kws_model.cc — TFLite-Micro interpreter wiring for the int8 DS-CNN model.
//
// Op set is intentionally minimal (MicroMutableOpResolver, not the "all
// ops" resolver) — this is what keeps flash usage down; add ops here only
// if you change train.py's architecture and the interpreter logs an
// "unsupported op" error at boot.

#include "kws_model.h"

#include <cmath>

#include "esp_log.h"
#include "mfcc.h"
#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "kws_model";

// Tensor arena: scratch RAM for activations. 40KB comfortably fits this
// tiny DS-CNN's largest intermediate tensor; bump it (and re-check the
// 256KB total RAM budget in README.md) if you enlarge the model.
constexpr int kTensorArenaSize = 40 * 1024;
alignas(16) static uint8_t s_tensor_arena[kTensorArenaSize];

// Ops used by train.py's build_model(): Conv2D, DepthwiseConv2D, BatchNorm
// (folded into Conv/DWConv at conversion time), ReLU, Mean (GlobalAvgPool),
// FullyConnected (Dense), Softmax, plus (De)Quantize at the int8 boundary.
static tflite::MicroMutableOpResolver<8> s_resolver;
static const tflite::Model *s_model = nullptr;
static tflite::MicroInterpreter *s_interpreter = nullptr;
static TfLiteTensor *s_input = nullptr;
static TfLiteTensor *s_output = nullptr;

bool kws_model_init(void) {
    if (g_naad_kws_model_data_len == 0) {
        ESP_LOGE(TAG, "model_data.cc is still the placeholder — run "
                      "training/train.py then training/convert.py, which "
                      "regenerate main/model_data.{h,cc} with your real model.");
        return false;
    }

    s_model = tflite::GetModel(g_naad_kws_model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %ld != supported %d",
                 (long)s_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    s_resolver.AddConv2D();
    s_resolver.AddDepthwiseConv2D();
    s_resolver.AddRelu();
    s_resolver.AddMean();
    s_resolver.AddFullyConnected();
    s_resolver.AddSoftmax();
    s_resolver.AddQuantize();
    s_resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(
        s_model, s_resolver, s_tensor_arena, kTensorArenaSize);
    s_interpreter = &static_interpreter;

    TfLiteStatus alloc_status = s_interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return false;
    }

    s_input = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    ESP_LOGI(TAG, "TFLite-Micro ready. Arena used: %d / %d bytes",
             (int)s_interpreter->arena_used_bytes(), kTensorArenaSize);
    return true;
}

float kws_model_infer(const float *mfcc_features) {
    const float in_scale = s_input->params.scale;
    const int in_zero_point = s_input->params.zero_point;

    int8_t *input_data = s_input->data.int8;
    const int n = MFCC_NUM_FRAMES * MFCC_NUM_COEFFS;
    for (int i = 0; i < n; i++) {
        float normalized = (mfcc_features[i] - MFCC_FEATURE_MEAN) / MFCC_FEATURE_STD;
        int32_t q = (int32_t)lrintf(normalized / in_scale) + in_zero_point;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        input_data[i] = (int8_t)q;
    }

    if (s_interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGW(TAG, "Invoke() failed");
        return 0.0f;
    }

    // Output layout matches config.LABELS = ["negative", "naad"]; index 1
    // is the keyword class.
    const float out_scale = s_output->params.scale;
    const int out_zero_point = s_output->params.zero_point;
    int8_t naad_q = s_output->data.int8[1];
    float naad_prob = (naad_q - out_zero_point) * out_scale;
    return naad_prob;
}
