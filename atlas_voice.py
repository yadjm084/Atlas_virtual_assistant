"""Audio, verification, wake-word, and ASR helpers for Atlas."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import whisper
from huggingface_hub import snapshot_download
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def get_dataset_root(base_dir: Path, repo_id: str) -> Path:
    """Download the voice dataset repo locally."""
    try:
        local_repo_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
        local_repo_path = Path(local_repo_path)
        print("BASE_DIR =", base_dir)
        print("Downloaded dataset repo to:", local_repo_path)
        return local_repo_path
    except Exception as e:
        print("Dataset download failed:", e)
        return base_dir / "missing_dataset_dir"


def get_enrollment_dir(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "enrollment",
        dataset_root / "user_verification" / "enrollment",
    ]
    for path in candidates:
        if path.exists():
            print("ENROLLMENT_DIR exists =", path)
            return path
    return candidates[0]


def get_wake_weights_path(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "wake_word.weights.h5",
        dataset_root / "models" / "wake_word.weights.h5",
        dataset_root / "wake_word" / "wake_word.weights.h5",
        dataset_root / "wake_word_model.weights.h5",
    ]
    for path in candidates:
        if path.exists():
            print("WAKE_WEIGHTS_PATH exists =", path)
            return path

    recursive_matches = list(dataset_root.rglob("wake_word.weights.h5"))
    if recursive_matches:
        print("WAKE_WEIGHTS_PATH found recursively =", recursive_matches[0])
        return recursive_matches[0]

    return candidates[0]


def load_and_preprocess_audio(file_path, target_sr=16000, target_length=40000):
    signal, sr = librosa.load(file_path, sr=target_sr)
    if len(signal) > target_length:
        signal = signal[:target_length]
    elif len(signal) < target_length:
        pad_amount = target_length - len(signal)
        signal = np.pad(signal, (0, pad_amount), mode="constant")
    return signal, sr


def extract_mfcc(signal, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)


def mfcc_to_fixed_vector(mfcc):
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_means, mfcc_stds])


def extract_speaker_vector(file_path, target_sr=16000, target_length=40000, n_mfcc=13):
    signal, sr = load_and_preprocess_audio(
        file_path=file_path,
        target_sr=target_sr,
        target_length=target_length,
    )
    mfcc = extract_mfcc(signal, sr, n_mfcc=n_mfcc)
    return mfcc_to_fixed_vector(mfcc)


def find_audio_files(folder_path: Path):
    wav_files = list(folder_path.glob("*.wav"))
    m4a_files = list(folder_path.glob("*.m4a"))
    return wav_files + m4a_files


def load_enrollment_profiles_with_normalization(
    enrollment_dir: Path,
    target_sr: int,
    target_length: int,
    n_mfcc: int,
):
    raw_rows = []
    messages = []

    if not enrollment_dir.exists():
        return {}, None, f"Enrollment directory not found: {enrollment_dir}"

    for speaker_folder in enrollment_dir.iterdir():
        if not speaker_folder.is_dir():
            continue

        speaker = speaker_folder.name
        audio_files = find_audio_files(speaker_folder)

        if len(audio_files) == 0:
            messages.append(f"{speaker}: no audio files found")
            continue

        loaded_count = 0
        for audio_file in audio_files:
            try:
                vector = extract_speaker_vector(
                    file_path=str(audio_file),
                    target_sr=target_sr,
                    target_length=target_length,
                    n_mfcc=n_mfcc,
                )
                raw_rows.append({"speaker": speaker, "filename": audio_file.name, "vector": vector})
                loaded_count += 1
            except Exception as e:
                messages.append(f"{speaker}: failed on {audio_file.name} ({e})")

        messages.append(f"{speaker}: loaded {loaded_count} enrollment files")

    if len(raw_rows) == 0:
        return {}, None, "No speaker profiles could be built."

    enrollment_matrix = np.vstack([row["vector"] for row in raw_rows])
    scaler = StandardScaler()
    scaler.fit(enrollment_matrix)

    for row in raw_rows:
        row["vector_scaled"] = scaler.transform(row["vector"].reshape(1, -1))[0]

    speaker_profiles = {}
    speakers = sorted(list(set(row["speaker"] for row in raw_rows)))
    for speaker in speakers:
        speaker_vectors = [row["vector_scaled"] for row in raw_rows if row["speaker"] == speaker]
        speaker_profiles[speaker] = np.mean(np.vstack(speaker_vectors), axis=0)

    return speaker_profiles, scaler, " | ".join(messages)


def compare_to_profiles(test_vector, speaker_profiles):
    scores = {}
    test_vector_2d = test_vector.reshape(1, -1)
    for speaker, profile_vector in speaker_profiles.items():
        profile_vector_2d = profile_vector.reshape(1, -1)
        scores[speaker] = float(cosine_similarity(test_vector_2d, profile_vector_2d)[0][0])
    return scores


def verify_audio_file(audio_path, speaker_profiles, scaler, target_sr, target_length, n_mfcc, threshold=0.5):
    test_vector = extract_speaker_vector(
        file_path=audio_path,
        target_sr=target_sr,
        target_length=target_length,
        n_mfcc=n_mfcc,
    )
    test_vector_scaled = scaler.transform(test_vector.reshape(1, -1))[0]
    scores = compare_to_profiles(test_vector_scaled, speaker_profiles)
    best_speaker = max(scores, key=scores.get)
    best_score = scores[best_speaker]
    accepted = best_score >= threshold
    predicted_user = best_speaker if accepted else "unknown"
    return {
        "accepted": accepted,
        "predicted_user": predicted_user,
        "best_score": best_score,
        "all_scores": scores,
    }


def build_wake_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(13, 63, 1)),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


def load_wake_model(weights_path: Path):
    try:
        if not weights_path.exists():
            return None, f"Wake word weights not found: {weights_path}"
        model = build_wake_model()
        model.load_weights(weights_path)
        return model, f"Wake word weights loaded from {weights_path.name}"
    except Exception as e:
        return None, f"Wake word model failed to load: {e}"


def predict_wake_word(audio_path, model, target_sr=16000, target_length=32000, n_mfcc=13, threshold=0.5):
    signal, sr = load_and_preprocess_audio(file_path=audio_path, target_sr=target_sr, target_length=target_length)
    mfcc = extract_mfcc(signal, sr, n_mfcc=n_mfcc)
    x_input = np.expand_dims(mfcc, axis=(0, -1))
    prob = float(model.predict(x_input, verbose=0)[0][0])
    pred = 1 if prob >= threshold else 0
    return {
        "probability_positive": prob,
        "predicted_label": pred,
        "predicted_name": "positive" if pred == 1 else "negative",
        "wake_detected": pred == 1,
    }


def load_asr_model(model_name="tiny"):
    try:
        model = whisper.load_model(model_name)
        return model, f"ASR model loaded: {model_name}"
    except Exception as e:
        return None, f"ASR model failed: {e}"


def transcribe_with_whisper(audio_path, model):
    result = model.transcribe(audio_path)
    return result["text"].strip()
