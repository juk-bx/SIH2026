"""
audio_utils.py — loading, augmentation and MFCC feature extraction.

The MFCC implementation intentionally uses plain numpy/librosa calls whose
math (mel filterbank + DCT-II) mirrors what we hand-roll in C on the ESP32
(main/mfcc.c). Keeping both sides doing the *same* transform is the single
most important thing for a quantized model to generalize from PC-recorded
training clips to on-device audio.
"""

import glob
import os
import random

import librosa
import numpy as np

from config import (
    AUG_NOISE_MAX_SNR_DB,
    AUG_NOISE_MIN_SNR_DB,
    AUG_NOISE_PROB,
    AUG_TIME_SHIFT_MS,
    AUG_VOLUME_MAX,
    AUG_VOLUME_MIN,
    CLIP_LEN,
    NUM_MEL_BINS,
    NUM_MFCC,
    SAMPLE_RATE,
    WINDOW_SIZE_SAMPLES,
    WINDOW_STRIDE_SAMPLES,
)


def list_wavs(directory, recursive=True):
    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted(glob.glob(os.path.join(directory, pattern), recursive=recursive))


def load_wav(path, target_len=CLIP_LEN):
    """Load a wav file, force mono/16k, and pad or center-crop to target_len."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < target_len:
        pad_total = target_len - len(audio)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode="constant")
    elif len(audio) > target_len:
        # Center crop. Works well as long as clips are already reasonably
        # trimmed to ~1s (see README.md recording guidelines).
        start = (len(audio) - target_len) // 2
        audio = audio[start:start + target_len]
    return audio.astype(np.float32)


def random_time_shift(audio, max_shift_ms=AUG_TIME_SHIFT_MS):
    max_shift = int(SAMPLE_RATE * max_shift_ms / 1000)
    shift = random.randint(-max_shift, max_shift)
    return np.roll(audio, shift)


def random_volume(audio):
    gain = random.uniform(AUG_VOLUME_MIN, AUG_VOLUME_MAX)
    return audio * gain


def mix_in_noise(audio, noise_clips):
    if not noise_clips or random.random() > AUG_NOISE_PROB:
        return audio
    noise = random.choice(noise_clips)
    if len(noise) < len(audio):
        reps = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, reps)
    start = random.randint(0, len(noise) - len(audio))
    noise_seg = noise[start:start + len(audio)]

    snr_db = random.uniform(AUG_NOISE_MIN_SNR_DB, AUG_NOISE_MAX_SNR_DB)
    sig_power = np.mean(audio ** 2) + 1e-9
    noise_power = np.mean(noise_seg ** 2) + 1e-9
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise_seg = noise_seg * np.sqrt(target_noise_power / noise_power)

    return audio + noise_seg


def augment(audio, noise_clips):
    audio = random_time_shift(audio)
    audio = random_volume(audio)
    audio = mix_in_noise(audio, noise_clips)
    return np.clip(audio, -1.0, 1.0)


def extract_mfcc(audio):
    """
    Returns an array of shape (NUM_FRAMES, NUM_MFCC) — float32, NOT yet
    quantized. train.py normalizes these before feeding the model.
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=NUM_MFCC,
        n_mels=NUM_MEL_BINS,
        n_fft=512,
        hop_length=WINDOW_STRIDE_SAMPLES,
        win_length=WINDOW_SIZE_SAMPLES,
        window="hann",
        center=False,
    )
    return mfcc.T.astype(np.float32)  # (frames, coeffs)


def load_noise_clips(directory):
    clips = []
    for p in list_wavs(directory):
        try:
            audio, _ = librosa.load(p, sr=SAMPLE_RATE, mono=True)
            if len(audio) > SAMPLE_RATE // 2:  # ignore very short files
                clips.append(audio.astype(np.float32))
        except Exception as e:
            print(f"  [warn] could not load noise clip {p}: {e}")
    return clips
