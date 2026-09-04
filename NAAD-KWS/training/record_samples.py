"""
record_samples.py — quick helper to record 1-second 16kHz mono wav clips
from your computer's microphone for building the dataset. Purely a
convenience tool; skip it if you already have recordings to upload.

Usage:
    python record_samples.py --label naad --count 50
    python record_samples.py --label negative --count 100
    python record_samples.py --label negative/background_noise --count 10 --duration 10

Press Enter before each clip to start recording.
"""

import argparse
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import DATASET_DIR, SAMPLE_RATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True,
                         help="'naad', 'negative', or 'negative/background_noise'")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--duration", type=float, default=1.0,
                         help="seconds per clip (use ~10s for background_noise)")
    args = parser.parse_args()

    out_dir = os.path.join(DATASET_DIR, args.label)
    os.makedirs(out_dir, exist_ok=True)
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".wav")])

    print(f"Recording {args.count} clips of {args.duration}s into {out_dir}")
    print("Press Enter to record each clip, Ctrl+C to stop early.\n")

    for i in range(args.count):
        idx = existing + i + 1
        input(f"[{i + 1}/{args.count}] Press Enter, then speak in 1s...")
        time.sleep(1)
        audio = sd.rec(int(args.duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                        channels=1, dtype="float32")
        sd.wait()
        path = os.path.join(out_dir, f"{os.path.basename(args.label)}_{idx:04d}.wav")
        sf.write(path, audio, SAMPLE_RATE)
        peak = float(np.max(np.abs(audio)))
        print(f"  saved {path}  (peak level: {peak:.2f}"
              f"{'  <- quiet, check mic gain' if peak < 0.05 else ''})")

    print("\nDone. Vary distance, angle and background noise across takes for robustness.")


if __name__ == "__main__":
    main()
