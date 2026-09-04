"""
try_single_clip.py — the fast "does it actually recognize ME saying it"
sanity check, before you ever touch the ESP32.

Usage:
    # record yourself (or use any voice recorder app / phone) and save a
    # ~1 second wav, then:
    python try_single_clip.py path/to/my_test.wav

    # or record straight from your mic right now (needs sounddevice):
    python try_single_clip.py --record

    # test the float model instead of the quantized one (before convert.py
    # has been run, or to compare float vs int8 accuracy):
    python try_single_clip.py my_test.wav --float
"""

import argparse
import os
import tempfile
import time

import numpy as np
import soundfile as sf
import tensorflow as tf

from audio_utils import extract_mfcc, load_wav
from config import (
    CLIP_LEN,
    DETECTION_THRESHOLD,
    LABELS,
    MODEL_OUT_DIR,
    SAMPLE_RATE,
    TFLITE_FLOAT_PATH,
    TFLITE_INT8_PATH,
)


def record_clip(duration_s=1.5):
    import sounddevice as sd

    print(f"Recording {duration_s}s... speak now!")
    time.sleep(0.3)
    audio = sd.rec(int(duration_s * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                    channels=1, dtype="float32")
    sd.wait()
    print("Done recording.")

    tmp_path = os.path.join(tempfile.gettempdir(), "naad_test_recording.wav")
    sf.write(tmp_path, audio, SAMPLE_RATE)
    return tmp_path


def predict(wav_path, tflite_path, mean, std):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    audio = load_wav(wav_path, target_len=CLIP_LEN)
    feats = extract_mfcc(audio)
    feats = (feats - mean) / (std + 1e-8)

    is_int8 = input_detail["dtype"] == np.int8
    if is_int8:
        scale, zero_point = input_detail["quantization"]
        x = np.round(feats / scale + zero_point).astype(np.int8)
    else:
        x = feats.astype(np.float32)
    x = x[np.newaxis, ..., np.newaxis]

    interpreter.set_tensor(input_detail["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(output_detail["index"])[0]

    if output_detail["dtype"] == np.int8:
        out_scale, out_zero_point = output_detail["quantization"]
        out = (out.astype(np.float32) - out_zero_point) * out_scale

    return out  # [p(negative), p(naad)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", nargs="?", help="path to a wav file to test")
    parser.add_argument("--record", action="store_true",
                         help="record a fresh clip from your mic instead")
    parser.add_argument("--float", action="store_true",
                         help="use the float32 model instead of the int8 one")
    args = parser.parse_args()

    if not args.wav and not args.record:
        parser.error("pass a wav path, or use --record to capture one live")

    tflite_path = TFLITE_FLOAT_PATH if args.float else TFLITE_INT8_PATH
    if not os.path.exists(tflite_path):
        raise SystemExit(
            f"{tflite_path} not found. Run train.py"
            + ("" if args.float else " then convert.py") + " first."
        )

    mean = float(np.load(os.path.join(MODEL_OUT_DIR, "feature_mean.npy")))
    std = float(np.load(os.path.join(MODEL_OUT_DIR, "feature_std.npy")))

    wav_path = record_clip() if args.record else args.wav

    probs = predict(wav_path, tflite_path, mean, std)

    print(f"\nFile: {wav_path}")
    print(f"Model: {tflite_path}")
    for label, p in zip(LABELS, probs):
        bar = "#" * int(p * 40)
        print(f"  {label:>9}: {p:.3f}  {bar}")

    naad_prob = probs[LABELS.index("naad")]
    if naad_prob >= DETECTION_THRESHOLD:
        print(f"\n>>> DETECTED as 'naad' (p={naad_prob:.3f} >= "
              f"threshold {DETECTION_THRESHOLD})")
    else:
        print(f"\n>>> NOT detected (p={naad_prob:.3f} < "
              f"threshold {DETECTION_THRESHOLD})")


if __name__ == "__main__":
    main()
