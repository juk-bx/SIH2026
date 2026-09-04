"""
config.py — single source of truth for every constant that must match
between the Python training pipeline and the ESP32 C firmware.

If you change anything here, re-run convert.py so it regenerates
esp32/naad_kws/main/model_data.{h,cc} AND double check
esp32/naad_kws/main/mfcc.h picks up the same numbers (they are mirrored
by hand there — see the comment block at the top of mfcc.h).
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
POS_DIR = os.path.join(DATASET_DIR, "naad")
NEG_DIR = os.path.join(DATASET_DIR, "negative")
NEG_NOISE_DIR = os.path.join(NEG_DIR, "background_noise")

MODEL_OUT_DIR = os.path.join(ROOT_DIR, "training", "artifacts")
KERAS_MODEL_PATH = os.path.join(MODEL_OUT_DIR, "naad_kws.keras")
TFLITE_FLOAT_PATH = os.path.join(MODEL_OUT_DIR, "naad_kws_float.tflite")
TFLITE_INT8_PATH = os.path.join(MODEL_OUT_DIR, "naad_kws_int8.tflite")

ESP32_MAIN_DIR = os.path.join(ROOT_DIR, "esp32", "naad_kws", "main")
MODEL_DATA_H = os.path.join(ESP32_MAIN_DIR, "model_data.h")
MODEL_DATA_CC = os.path.join(ESP32_MAIN_DIR, "model_data.cc")

# ---------------------------------------------------------------------------
# Audio / feature-extraction parameters
# These MUST be identical on-device (see main/mfcc.h) or the quantized
# model will see a distribution shift and accuracy will collapse.
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000            # Hz
CLIP_DURATION_MS = 1000        # each training/inference window is 1.0 s
CLIP_LEN = SAMPLE_RATE * CLIP_DURATION_MS // 1000   # 16000 samples

WINDOW_SIZE_MS = 30            # MFCC analysis frame length
WINDOW_STRIDE_MS = 20          # MFCC hop length
NUM_MFCC = 13                  # coefficients per frame (incl. C0)
NUM_MEL_BINS = 26              # mel filterbank channels feeding the DCT
FFT_LEN = 512                  # next pow2 >= window_size_samples (480)

WINDOW_SIZE_SAMPLES = SAMPLE_RATE * WINDOW_SIZE_MS // 1000     # 480
WINDOW_STRIDE_SAMPLES = SAMPLE_RATE * WINDOW_STRIDE_MS // 1000  # 320

# number of MFCC frames in one 1-second clip
NUM_FRAMES = 1 + (CLIP_LEN - WINDOW_SIZE_SAMPLES) // WINDOW_STRIDE_SAMPLES  # 49

INPUT_SHAPE = (NUM_FRAMES, NUM_MFCC, 1)   # (49, 13, 1) -> 637 int8 = 637 B input tensor

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABELS = ["negative", "naad"]   # index 0 = negative/background, index 1 = keyword
NUM_CLASSES = len(LABELS)

# Sliding-window inference on-device: how often we run a fresh classification
# while continuously listening. 200 ms hop = 5 inferences/sec, cheap on an
# int8 DS-CNN of this size (<15 ms per inference on the ESP32 core).
STREAM_HOP_MS = 200

# ---------------------------------------------------------------------------
# Decision thresholds (tune with test_model.py against your own dataset)
# ---------------------------------------------------------------------------
DETECTION_THRESHOLD = 0.85     # softmax prob for class "naad" to count as a hit
DETECTION_HANGOVER_WINDOWS = 3  # consecutive windows required before firing
                                 # (debounce -> kills single-frame false positives)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Data augmentation
AUG_TIME_SHIFT_MS = 100          # random shift +/- this many ms
AUG_NOISE_PROB = 0.8             # probability of mixing in background noise
AUG_NOISE_MIN_SNR_DB = 3
AUG_NOISE_MAX_SNR_DB = 15
AUG_VOLUME_MIN = 0.7
AUG_VOLUME_MAX = 1.2
