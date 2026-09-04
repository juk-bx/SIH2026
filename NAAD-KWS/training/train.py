"""
train.py — train the "naad" keyword-spotting model.

Usage:
    python train.py

Expects:
    dataset/naad/*.wav                    -> positive (keyword) clips, ~1s each
    dataset/negative/*.wav                -> negative speech / other words
    dataset/negative/background_noise/*.wav -> long ambient noise clips used
                                                only for augmentation (mixed
                                                into both classes at random
                                                SNR), NOT used as standalone
                                                training examples unless you
                                                also chop them into 1s clips
                                                and drop them in dataset/negative.

Outputs (under training/artifacts/):
    naad_kws.keras            - full precision Keras model
    naad_kws_float.tflite     - float32 TFLite (sanity check only)
    training_history.png      - loss/accuracy curves
    confusion_matrix.png      - held-out test set confusion matrix

Run convert.py afterwards to produce the int8 TFLite model + the
model_data.h/.cc pair that get compiled into the ESP32 firmware.
"""

import os
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from audio_utils import augment, extract_mfcc, list_wavs, load_noise_clips, load_wav
from config import (
    BATCH_SIZE,
    EPOCHS,
    INPUT_SHAPE,
    KERAS_MODEL_PATH,
    LABELS,
    LEARNING_RATE,
    MODEL_OUT_DIR,
    NEG_DIR,
    NEG_NOISE_DIR,
    NUM_CLASSES,
    POS_DIR,
    RANDOM_SEED,
    TEST_SPLIT,
    TFLITE_FLOAT_PATH,
    VAL_SPLIT,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 1. Build the dataset
# ---------------------------------------------------------------------------
def build_dataset():
    pos_files = list_wavs(POS_DIR, recursive=False)
    neg_files = [f for f in list_wavs(NEG_DIR, recursive=True)
                 if os.path.dirname(f) != NEG_NOISE_DIR]

    if len(pos_files) == 0:
        raise RuntimeError(
            f"No wav files found in {POS_DIR}. Upload keyword recordings there "
            f"before running train.py."
        )
    if len(neg_files) == 0:
        raise RuntimeError(
            f"No wav files found in {NEG_DIR} (excluding background_noise/). "
            f"You need negative examples: other spoken words, silence, "
            f"and ambient noise segments."
        )

    print(f"Found {len(pos_files)} positive ('naad') clips")
    print(f"Found {len(neg_files)} negative clips")

    noise_clips = load_noise_clips(NEG_NOISE_DIR)
    print(f"Loaded {len(noise_clips)} background-noise clips for augmentation")

    if len(pos_files) < 50:
        print("\n[WARNING] Fewer than 50 positive clips found. A robust KWS "
              "model typically needs 200-500+ utterances of the keyword "
              "recorded by multiple speakers, distances and rooms. Consider "
              "collecting more data before trusting these results.\n")

    return pos_files, neg_files, noise_clips


def clips_to_features(files, label, noise_clips, augment_factor):
    """
    For each wav file: extract 1 clean (unaugmented) example, plus
    `augment_factor` further augmented examples (0 for validation/test).
    """
    X, y = [], []
    for f in files:
        try:
            audio = load_wav(f)
        except Exception as e:
            print(f"  [warn] skipping unreadable file {f}: {e}")
            continue
        X.append(extract_mfcc(audio))
        y.append(label)
        for _ in range(augment_factor):
            aug_audio = augment(audio, noise_clips)
            X.append(extract_mfcc(aug_audio))
            y.append(label)
    return X, y


def normalize_features(X, mean, std):
    return (X - mean) / (std + 1e-8)


# ---------------------------------------------------------------------------
# 2. Model: tiny DS-CNN (depthwise-separable CNN)
#    ~15-25K parameters -> a few tens of KB once int8 quantized, well inside
#    a 256KB RAM / small flash budget on an ESP32.
# ---------------------------------------------------------------------------
def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    inputs = tf.keras.Input(shape=input_shape, name="mfcc_input")

    x = tf.keras.layers.Conv2D(8, (10, 4), strides=(2, 2), padding="same",
                                use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for filters in (16, 16):
        x = tf.keras.layers.DepthwiseConv2D((3, 3), padding="same",
                                              use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, (1, 1), padding="same",
                                    use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="naad_ds_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)

    pos_files, neg_files, noise_clips = build_dataset()

    pos_train_f, pos_temp_f = train_test_split(
        pos_files, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=RANDOM_SEED)
    pos_val_f, pos_test_f = train_test_split(
        pos_temp_f, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
        random_state=RANDOM_SEED)

    neg_train_f, neg_temp_f = train_test_split(
        neg_files, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=RANDOM_SEED)
    neg_val_f, neg_test_f = train_test_split(
        neg_temp_f, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
        random_state=RANDOM_SEED)

    print("\nExtracting features (this can take a while the first run)...")
    # Heavier augmentation on the positive class since it's usually the
    # smaller, harder-to-collect one.
    Xp_tr, yp_tr = clips_to_features(pos_train_f, 1, noise_clips, augment_factor=6)
    Xn_tr, yn_tr = clips_to_features(neg_train_f, 0, noise_clips, augment_factor=2)
    Xp_val, yp_val = clips_to_features(pos_val_f, 1, noise_clips, augment_factor=0)
    Xn_val, yn_val = clips_to_features(neg_val_f, 0, noise_clips, augment_factor=0)
    Xp_te, yp_te = clips_to_features(pos_test_f, 1, noise_clips, augment_factor=0)
    Xn_te, yn_te = clips_to_features(neg_test_f, 0, noise_clips, augment_factor=0)

    X_train = np.array(Xp_tr + Xn_tr, dtype=np.float32)
    y_train = np.array(yp_tr + yn_tr, dtype=np.int32)
    X_val = np.array(Xp_val + Xn_val, dtype=np.float32)
    y_val = np.array(yp_val + yn_val, dtype=np.int32)
    X_test = np.array(Xp_te + Xn_te, dtype=np.float32)
    y_test = np.array(yp_te + yn_te, dtype=np.int32)

    # Normalize using training-set statistics only; save them, since
    # convert.py needs the *same* stats to build the int8 quantization
    # representative dataset and main.c needs them (baked in) on-device.
    mean = X_train.mean()
    std = X_train.std()
    np.save(os.path.join(MODEL_OUT_DIR, "feature_mean.npy"), mean)
    np.save(os.path.join(MODEL_OUT_DIR, "feature_std.npy"), std)
    print(f"Feature normalization: mean={mean:.4f} std={std:.4f}")

    X_train = normalize_features(X_train, mean, std)[..., np.newaxis]
    X_val = normalize_features(X_val, mean, std)[..., np.newaxis]
    X_test = normalize_features(X_test, mean, std)[..., np.newaxis]

    print(f"\nDataset sizes -> train: {len(X_train)}  val: {len(X_val)}  "
          f"test: {len(X_test)}")

    # Class weighting in case pos/neg counts are imbalanced after augmentation
    n_pos, n_neg = int(y_train.sum()), int(len(y_train) - y_train.sum())
    total = n_pos + n_neg
    class_weight = {
        0: total / (2 * max(n_neg, 1)),
        1: total / (2 * max(n_pos, 1)),
    }
    print(f"Class weights: {class_weight}")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # ---- Evaluate on held-out test set -----------------------------------
    print("\nEvaluating on held-out test set...")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    print(classification_report(y_test, y_pred, target_names=LABELS))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    false_accepts = cm[0][1] if cm.shape == (2, 2) else None
    if false_accepts is not None and (cm[0].sum() > 0):
        far = false_accepts / cm[0].sum()
        print(f"False Accept Rate on test negatives: {far * 100:.2f}%")

    # ---- Save artifacts -----------------------------------------------
    model.save(KERAS_MODEL_PATH)
    print(f"\nSaved Keras model -> {KERAS_MODEL_PATH}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float = converter.convert()
    with open(TFLITE_FLOAT_PATH, "wb") as f:
        f.write(tflite_float)
    print(f"Saved float32 TFLite (sanity check) -> {TFLITE_FLOAT_PATH}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history.history["loss"], label="train")
        axes[0].plot(history.history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(history.history["accuracy"], label="train")
        axes[1].plot(history.history["val_accuracy"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(MODEL_OUT_DIR, "training_history.png"), dpi=120)
        print("Saved training_history.png")
    except ImportError:
        print("matplotlib not installed, skipping training curve plot")

    print("\nNext step: python convert.py   (produces the int8 TFLite model "
          "and model_data.h/.cc for the ESP32 build)")


if __name__ == "__main__":
    main()
