// kws_model.h — thin C-callable wrapper around the C++-only TFLite-Micro
// API, so main.c can stay plain C.
#ifndef NAAD_KWS_MODEL_H_
#define NAAD_KWS_MODEL_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Allocates the tensor arena, builds the op resolver/interpreter, and
// checks the embedded model against the TFLite schema version. Returns
// false (and logs why) if anything fails — callers should treat that as
// fatal for KWS but can still let the device boot for OTA/debug purposes.
bool kws_model_init(void);

// Runs inference on NUM_FRAMES x NUM_MFCC MFCC coefficients already laid
// out row-major float32 in `mfcc_features` (NUM_FRAMES*NUM_MFCC elements).
// Handles float->int8 quantization internally using the model's own input
// scale/zero-point. Returns the softmax probability of the "naad" class.
float kws_model_infer(const float *mfcc_features);

#ifdef __cplusplus
}
#endif

#endif  // NAAD_KWS_MODEL_H_
