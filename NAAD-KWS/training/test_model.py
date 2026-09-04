"""
test_model.py — evaluate the QUANTIZED (int8) TFLite model exactly as it
will run on-device, and report the three metrics the problem statement
asks for:

  * Accuracy      -> true-positive rate on held-out "naad" clips
  * False accepts -> how often negative audio is misclassified as "naad"
  * Latency       -> wall-clock time per inference call (proxy for
                      on-MCU latency; the ESP32 will be slower but this
                      catches egregiously oversized models early)

Usage:
    python test_model.py [--tflite path/to/model.tflite] [--all]

--all runs over the ENTIRE dataset (not just a held-out split) — useful as
a quick smoke test right after collecting new recordings.
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from audio_utils import extract_mfcc, list_wavs, load_wav
from config import (
    LABELS,
    MODEL_OUT_DIR,
    NEG_DIR,
    NEG_NOISE_DIR,
    POS_DIR,
    TFLITE_INT8_PATH,
)


def load_interpreter(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


def quantize_input(feats, input_detail, mean, std):
    feats = (feats - mean) / (std + 1e-8)
    scale, zero_point = input_detail["quantization"]
    q = feats / scale + zero_point
    q = np.round(q).astype(np.int8)
    return q[np.newaxis, ..., np.newaxis]


def run_inference(interpreter, x_int8):
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    t0 = time.perf_counter()
    interpreter.set_tensor(input_detail["index"], x_int8)
    interpreter.invoke()
    out = interpreter.get_tensor(output_detail["index"])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out_scale, out_zero_point = output_detail["quantization"]
    probs = (out.astype(np.float32) - out_zero_point) * out_scale
    return probs[0], elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", default=TFLITE_INT8_PATH)
    parser.add_argument("--all", action="store_true",
                         help="evaluate on the full dataset instead of a split")
    args = parser.parse_args()

    if not os.path.exists(args.tflite):
        raise SystemExit(f"{args.tflite} not found — run train.py then convert.py first.")

    mean = float(np.load(os.path.join(MODEL_OUT_DIR, "feature_mean.npy")))
    std = float(np.load(os.path.join(MODEL_OUT_DIR, "feature_std.npy")))

    interpreter = load_interpreter(args.tflite)
    input_detail = interpreter.get_input_details()[0]

    pos_files = list_wavs(POS_DIR, recursive=False)
    neg_files = [f for f in list_wavs(NEG_DIR, recursive=True)
                 if os.path.dirname(f) != NEG_NOISE_DIR]

    if not args.all:
        # crude reproducible held-out slice matching train.py's proportions
        n_pos_test = max(1, int(len(pos_files) * 0.15))
        n_neg_test = max(1, int(len(neg_files) * 0.15))
        pos_files = pos_files[-n_pos_test:]
        neg_files = neg_files[-n_neg_test:]

    print(f"Evaluating on {len(pos_files)} positive / {len(neg_files)} negative clips")

    y_true, y_pred, latencies = [], [], []

    for label, files in ((1, pos_files), (0, neg_files)):
        for f in files:
            try:
                audio = load_wav(f)
            except Exception as e:
                print(f"  [warn] skipping {f}: {e}")
                continue
            feats = extract_mfcc(audio)
            x_int8 = quantize_input(feats, input_detail, mean, std)
            probs, elapsed_ms = run_inference(interpreter, x_int8)
            pred = int(np.argmax(probs))
            y_true.append(label)
            y_pred.append(pred)
            latencies.append(elapsed_ms)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion matrix (rows=true, cols=pred) [negative, naad]:")
    print(cm)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (fp + tn) if (fp + tn) else float("nan")
    print(f"\nTrue-Positive Rate (keyword correctly detected): {tpr * 100:.2f}%")
    print(f"False-Accept Rate  (negative wrongly triggers):   {far * 100:.2f}%")

    print(f"\nModel file size: {os.path.getsize(args.tflite) / 1024:.1f} KB")
    print(f"PC-side inference latency: mean={np.mean(latencies):.2f} ms, "
          f"p95={np.percentile(latencies, 95):.2f} ms  "
          f"(this is a lower bound / sanity check — measure the real figure "
          f"on the ESP32 with esp_timer around interpreter->Invoke() in main.c)")


if __name__ == "__main__":
    main()
