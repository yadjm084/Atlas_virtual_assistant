import gradio as gr
import json
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf
import whisper
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from huggingface_hub import snapshot_download


# =========================================================
# USER VERIFICATION + DATASET CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
HF_DATASET_REPO = "yadjm084/atlas-voice-data"

TARGET_SR = 16000
TARGET_DURATION = 2.5
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)
N_MFCC = 13

# Final threshold selected after normalization experiments
CHOSEN_THRESHOLD = 0.5

VALID_CODES = ["Adjmal", "Nair", "Sharma"]

# Wake word parameters
WAKE_TARGET_SR = 16000
WAKE_TARGET_DURATION = 2.0
WAKE_TARGET_LENGTH = int(WAKE_TARGET_SR * WAKE_TARGET_DURATION)
WAKE_N_MFCC = 13
WAKE_THRESHOLD = 0.5
WAKE_BYPASS_CODE = "Hey Atlas"


def get_dataset_root():
    """
    Download the public Hugging Face dataset repo locally
    and return the local dataset root path.
    """
    try:
        local_repo_path = snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset"
        )

        local_repo_path = Path(local_repo_path)

        print("BASE_DIR =", BASE_DIR)
        print("Downloaded dataset repo to:", local_repo_path)

        return local_repo_path

    except Exception as e:
        print("Dataset download failed:", e)
        return BASE_DIR / "missing_dataset_dir"


DATASET_ROOT = get_dataset_root()


def get_enrollment_dir(dataset_root):
    """
    Support either:
    - dataset_root/enrollment
    - dataset_root/user_verification/enrollment
    """
    candidates = [
        dataset_root / "enrollment",
        dataset_root / "user_verification" / "enrollment",
    ]

    for path in candidates:
        if path.exists():
            print("ENROLLMENT_DIR exists =", path)
            return path

    return candidates[0]


def get_wake_weights_path(dataset_root):
    """
    Search the dataset repo for wake-word weights.
    Supports a few common names and falls back to recursive search.
    """
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


ENROLLMENT_DIR = get_enrollment_dir(DATASET_ROOT)
WAKE_WEIGHTS_PATH = get_wake_weights_path(DATASET_ROOT)


# =========================================================
# AUDIO + FEATURE FUNCTIONS
# =========================================================

def load_and_preprocess_audio(file_path, target_sr=16000, target_length=40000):
    signal, sr = librosa.load(file_path, sr=target_sr)

    if len(signal) > target_length:
        signal = signal[:target_length]
    elif len(signal) < target_length:
        pad_amount = target_length - len(signal)
        signal = np.pad(signal, (0, pad_amount), mode="constant")

    return signal, sr


def extract_mfcc(signal, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc
    )
    return mfcc


def mfcc_to_fixed_vector(mfcc):
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)
    feature_vector = np.concatenate([mfcc_means, mfcc_stds])
    return feature_vector


def extract_speaker_vector(file_path, target_sr=16000, target_length=40000, n_mfcc=13):
    signal, sr = load_and_preprocess_audio(
        file_path=file_path,
        target_sr=target_sr,
        target_length=target_length
    )

    mfcc = extract_mfcc(signal, sr, n_mfcc=n_mfcc)
    feature_vector = mfcc_to_fixed_vector(mfcc)

    return feature_vector


# =========================================================
# PROFILE LOADING + NORMALIZATION
# =========================================================

def find_audio_files(folder_path):
    wav_files = list(folder_path.glob("*.wav"))
    m4a_files = list(folder_path.glob("*.m4a"))
    return wav_files + m4a_files


def load_enrollment_profiles_with_normalization(enrollment_dir):
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
                    target_sr=TARGET_SR,
                    target_length=TARGET_LENGTH,
                    n_mfcc=N_MFCC
                )

                raw_rows.append({
                    "speaker": speaker,
                    "filename": audio_file.name,
                    "vector": vector
                })
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
        mean_profile = np.mean(np.vstack(speaker_vectors), axis=0)
        speaker_profiles[speaker] = mean_profile

    return speaker_profiles, scaler, " | ".join(messages)


def compare_to_profiles(test_vector, speaker_profiles):
    scores = {}
    test_vector_2d = test_vector.reshape(1, -1)

    for speaker, profile_vector in speaker_profiles.items():
        profile_vector_2d = profile_vector.reshape(1, -1)
        score = cosine_similarity(test_vector_2d, profile_vector_2d)[0][0]
        scores[speaker] = float(score)

    return scores


def verify_audio_file(audio_path, speaker_profiles, scaler, threshold=0.5):
    test_vector = extract_speaker_vector(
        file_path=audio_path,
        target_sr=TARGET_SR,
        target_length=TARGET_LENGTH,
        n_mfcc=N_MFCC
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
        "all_scores": scores
    }


SPEAKER_PROFILES, FEATURE_SCALER, PROFILE_LOAD_STATUS = load_enrollment_profiles_with_normalization(ENROLLMENT_DIR)


# =========================================================
# WAKE WORD MODEL LOADING + INFERENCE
# =========================================================

def build_wake_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(13, 63, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model


def load_wake_model(weights_path):
    try:
        if not weights_path.exists():
            return None, f"Wake word weights not found: {weights_path}"

        model = build_wake_model()
        model.load_weights(weights_path)

        return model, f"Wake word weights loaded from {weights_path.name}"
    except Exception as e:
        return None, f"Wake word model failed to load: {e}"


WAKE_MODEL, WAKE_MODEL_STATUS = load_wake_model(WAKE_WEIGHTS_PATH)


def load_whisper_model():
    try:
        model = whisper.load_model("tiny")
        return model, "Whisper tiny model loaded successfully"
    except Exception as e:
        return None, f"Whisper model failed to load: {e}"


WHISPER_MODEL, WHISPER_MODEL_STATUS = load_whisper_model()


def transcribe_with_whisper(audio_path, model):
    audio, _ = librosa.load(audio_path, sr=16000)
    result = model.transcribe(audio, fp16=False)
    return result["text"].strip()


def predict_wake_word(audio_path, model, target_sr=16000, target_length=32000, n_mfcc=13, threshold=0.5):
    signal, sr = load_and_preprocess_audio(
        file_path=audio_path,
        target_sr=target_sr,
        target_length=target_length
    )

    mfcc = extract_mfcc(signal, sr, n_mfcc=n_mfcc)
    x_input = np.expand_dims(mfcc, axis=(0, -1))  # (1, 13, time, 1)

    prob = float(model.predict(x_input, verbose=0)[0][0])
    pred = 1 if prob >= threshold else 0

    return {
        "probability_positive": prob,
        "predicted_label": pred,
        "predicted_name": "positive" if pred == 1 else "negative",
        "wake_detected": pred == 1
    }


# =========================================================
# STATE FUNCTIONS
# =========================================================

def init_state():
    return {
        "verified": False,
        "verified_user": "None",
        "awake": False,
        "transcript": "",
        "intent": "",
        "slots": {},
        "api_result": {},
        "answer_text": "",
        "verification_scores": {},
        "verification_best_score": None,
        "wake_probability": None,
        "control_state": {
            "lamp": "off",
            "temperature": 20
        }
    }


def get_status_text(state):
    verify_score_text = (
        f"{state['verification_best_score']:.4f}"
        if state["verification_best_score"] is not None
        else "None"
    )

    wake_score_text = (
        f"{state['wake_probability']:.4f}"
        if state["wake_probability"] is not None
        else "None"
    )

    return (
        f"Verified: {state['verified']}\n"
        f"Verified User: {state['verified_user']}\n"
        f"Best Verification Score: {verify_score_text}\n"
        f"Awake: {state['awake']}\n"
        f"Wake Probability: {wake_score_text}\n"
        f"Intent: {state['intent'] if state['intent'] else 'None'}"
    )


# =========================================================
# USER VERIFICATION FUNCTIONS
# =========================================================

def do_verify(audio, state):
    if audio is None:
        return "Please record or upload an audio file first.", "{}", state, get_status_text(state)

    if not SPEAKER_PROFILES or FEATURE_SCALER is None:
        return (
            f"Speaker profiles could not be loaded. {PROFILE_LOAD_STATUS}",
            "{}",
            state,
            get_status_text(state)
        )

    try:
        result = verify_audio_file(
            audio_path=audio,
            speaker_profiles=SPEAKER_PROFILES,
            scaler=FEATURE_SCALER,
            threshold=CHOSEN_THRESHOLD
        )

        state["verification_scores"] = result["all_scores"]
        state["verification_best_score"] = result["best_score"]

        if result["accepted"]:
            state["verified"] = True
            state["verified_user"] = result["predicted_user"]

            message = (
                f"Verification successful. "
                f"User identified as {result['predicted_user']} "
                f"(score={result['best_score']:.4f}, threshold={CHOSEN_THRESHOLD})."
            )
        else:
            state["verified"] = False
            state["verified_user"] = "None"
            state["awake"] = False
            state["wake_probability"] = None

            message = (
                f"Verification failed. Atlas remains locked "
                f"(best score={result['best_score']:.4f}, threshold={CHOSEN_THRESHOLD})."
            )

        scores_json = json.dumps(result["all_scores"], indent=2)
        return message, scores_json, state, get_status_text(state)

    except Exception as e:
        return f"Verification error: {e}", "{}", state, get_status_text(state)


def verify_with_code(code_input, state):
    code = code_input.strip()

    if code in VALID_CODES:
        state["verified"] = True
        state["verified_user"] = code
        state["verification_best_score"] = None
        state["verification_scores"] = {}

        return (
            f"Verification bypass successful. User set to {code}.",
            "{}",
            state,
            get_status_text(state)
        )

    return (
        "Invalid verification code. Use Adjmal, Nair, or Sharma.",
        "{}",
        state,
        get_status_text(state)
    )


def reset_verification(state):
    state["verified"] = False
    state["verified_user"] = "None"
    state["verification_scores"] = {}
    state["verification_best_score"] = None
    state["awake"] = False
    state["wake_probability"] = None
    state["transcript"] = ""

    return "", "", "{}", state, get_status_text(state)


# =========================================================
# WAKE WORD FUNCTIONS
# =========================================================

def do_wake(audio, state):
    if not state["verified"]:
        return "Please complete user verification first.", state, get_status_text(state)

    if audio is None:
        return "Please record or upload an audio file first.", state, get_status_text(state)

    if WAKE_MODEL is None:
        return f"Wake word model unavailable. {WAKE_MODEL_STATUS}", state, get_status_text(state)

    try:
        result = predict_wake_word(
            audio_path=audio,
            model=WAKE_MODEL,
            target_sr=WAKE_TARGET_SR,
            target_length=WAKE_TARGET_LENGTH,
            n_mfcc=WAKE_N_MFCC,
            threshold=WAKE_THRESHOLD
        )

        state["wake_probability"] = result["probability_positive"]

        if result["wake_detected"]:
            state["awake"] = True
            message = (
                f"Wake word detected. Atlas is now awake "
                f"(probability={result['probability_positive']:.4f}, threshold={WAKE_THRESHOLD})."
            )
        else:
            state["awake"] = False
            message = (
                f"Wake word not detected "
                f"(probability={result['probability_positive']:.4f}, threshold={WAKE_THRESHOLD})."
            )

        return message, state, get_status_text(state)

    except Exception as e:
        return f"Wake word error: {e}", state, get_status_text(state)


def skip_wake(state):
    if not state["verified"]:
        return "Please complete user verification first.", state, get_status_text(state)

    state["awake"] = True
    state["wake_probability"] = None
    return "Wake word detection skipped.", state, get_status_text(state)


def skip_wake_with_code(code_input, state):
    if not state["verified"]:
        return "Please complete user verification first.", state, get_status_text(state)

    code = code_input.strip()
    if code.lower() == WAKE_BYPASS_CODE.lower():
        state["awake"] = True
        state["wake_probability"] = None
        return "Wake word bypass successful. Atlas is now awake.", state, get_status_text(state)

    return f"Invalid wake word code. Use: {WAKE_BYPASS_CODE}", state, get_status_text(state)


def reset_wake_word(state):
    state["awake"] = False
    state["wake_probability"] = None
    state["transcript"] = ""
    state["intent"] = ""
    state["slots"] = {}
    state["api_result"] = {}
    state["answer_text"] = ""

    return "", "", "", "", "", state, get_status_text(state)


# =========================================================
# ASR FUNCTIONS
# =========================================================

def do_asr(audio, state):
    if not state["verified"]:
        return "Please complete user verification first.", state, get_status_text(state)

    if not state["awake"]:
        return "Please detect the wake word first.", state, get_status_text(state)

    fake_text = "turn on the lamp"
    state["transcript"] = fake_text
    return fake_text, state, get_status_text(state)


# =========================================================
# INTENT FUNCTIONS
# =========================================================

def do_intent(transcript, state):
    if not state["verified"]:
        return "Verification required.", "{}", state, get_status_text(state)

    if not state["awake"]:
        return "Wake word required.", "{}", state, get_status_text(state)

    if not transcript.strip():
        return "No transcript available.", "{}", state, get_status_text(state)

    fake_intent = "control_device"
    fake_slots = {
        "device": "lamp",
        "action": "on"
    }

    state["intent"] = fake_intent
    state["slots"] = fake_slots

    return (
        fake_intent,
        json.dumps(fake_slots, indent=2),
        state,
        get_status_text(state)
    )


def use_manual_intent(manual_intent, manual_slots, state):
    state["intent"] = manual_intent.strip()

    try:
        state["slots"] = json.loads(manual_slots) if manual_slots.strip() else {}

        return (
            state["intent"],
            json.dumps(state["slots"], indent=2),
            state,
            get_status_text(state)
        )
    except json.JSONDecodeError:
        return (
            state["intent"],
            "Invalid JSON format in manual slots.",
            state,
            get_status_text(state)
        )


# =========================================================
# FULFILLMENT FUNCTIONS
# =========================================================

def do_fulfillment(state):
    if not state["intent"]:
        return (
            "Please detect or enter an intent first.",
            json.dumps(state["control_state"], indent=2),
            state,
            get_status_text(state)
        )

    if state["intent"] == "control_device":
        device = state["slots"].get("device", "lamp")
        action = state["slots"].get("action", "on")

        if device == "lamp":
            state["control_state"]["lamp"] = action

        api_result = {
            "status": "success",
            "message": f"{device} turned {action}"
        }
    else:
        api_result = {
            "status": "success",
            "message": "Placeholder fulfillment completed."
        }

    state["api_result"] = api_result

    return (
        json.dumps(api_result, indent=2),
        json.dumps(state["control_state"], indent=2),
        state,
        get_status_text(state)
    )


def use_manual_api_result(manual_api_result, state):
    try:
        state["api_result"] = json.loads(manual_api_result) if manual_api_result.strip() else {}

        return (
            json.dumps(state["api_result"], indent=2),
            json.dumps(state["control_state"], indent=2),
            state,
            get_status_text(state)
        )
    except json.JSONDecodeError:
        return (
            "Invalid JSON format in manual API result.",
            json.dumps(state["control_state"], indent=2),
            state,
            get_status_text(state)
        )


# =========================================================
# ANSWER + TTS FUNCTIONS
# =========================================================

def do_answer(state):
    if state["api_result"]:
        answer = f"Assistant response: {state['api_result'].get('message', 'Done.')}"
    else:
        answer = "Assistant response: Placeholder answer generated."

    state["answer_text"] = answer
    return answer, state, get_status_text(state)


def do_tts(state):
    return f"TTS would say: {state['answer_text']}"


# =========================================================
# RESET FUNCTION
# =========================================================

def reset_all():
    state = init_state()

    return (
        state,
        get_status_text(state),
        "",
        "",
        "{}",
        "",
        "",
        "",
        "",
        "",
        json.dumps(state["control_state"], indent=2),
        "",
        "",
        "control_device",
        '{\n  "device": "lamp",\n  "action": "on"\n}',
        '{\n  "status": "success",\n  "message": "lamp turned on"\n}'
    )


# =========================================================
# GRADIO INTERFACE
# =========================================================

with gr.Blocks(title="Atlas - Virtual Assistant") as demo:
    gr.Markdown("# Atlas - Virtual Assistant")
    gr.Markdown("User verification uses normalized MFCC profiles. Wake word detection uses CNN weights from the HF dataset repo.")

    state = gr.State(init_state())

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Voice Input")

        assistant_status = gr.Textbox(
            label="Assistant Status",
            value=get_status_text(init_state()),
            lines=6
        )

        control_box = gr.Textbox(
            label="Control System State",
            value=json.dumps(init_state()["control_state"], indent=2),
            lines=8
        )

    with gr.Group():
        gr.Markdown("## User Verification")
        gr.Markdown(f"Enrollment status: {PROFILE_LOAD_STATUS}")
        gr.Markdown(f"Verification threshold: {CHOSEN_THRESHOLD}")

        verify_output = gr.Textbox(label="Verification Output", lines=3)

        verification_scores_output = gr.Textbox(
            label="Verification Scores",
            lines=8,
            value="{}"
        )

        with gr.Row():
            btn_verify = gr.Button("Run Verification", variant="primary")
            btn_reset_verification = gr.Button("Reset Verification")

        gr.Markdown("### Bypass Verification with Code")
        gr.Markdown("Allowed codes: `Adjmal`, `Nair`, `Sharma`")

        with gr.Row():
            verification_code_input = gr.Textbox(
                label="Verification Code",
                placeholder="Enter Adjmal, Nair, or Sharma"
            )
            btn_skip_verify = gr.Button("Skip Verification with Code")

    with gr.Group():
        gr.Markdown("## Wake Word")
        gr.Markdown(f"Wake word model status: {WAKE_MODEL_STATUS}")
        gr.Markdown(f"Wake word threshold: {WAKE_THRESHOLD}")

        wake_output = gr.Textbox(label="Wake Word Output", lines=2)

        with gr.Row():
            btn_wake = gr.Button("Run Wake Word")
            btn_skip_wake = gr.Button("Skip Wake Word")

    with gr.Group():
        gr.Markdown("## Speech Recognition")
        transcript_box = gr.Textbox(label="Transcript", lines=3)

        with gr.Row():
            btn_asr = gr.Button("Run ASR on Wake / Command Audio")

    with gr.Group():
        gr.Markdown("## Intent Detection")

        with gr.Row():
            intent_box = gr.Textbox(label="Detected Intent")
            slots_box = gr.Textbox(label="Detected Slots", lines=6)

        with gr.Accordion("Manual / Bypass Options for Intent", open=False):
            manual_intent = gr.Textbox(label="Manual Intent", value="control_device")

            manual_slots = gr.Textbox(
                label="Manual Slots (JSON)",
                lines=6,
                value="""{
  \"device\": \"lamp\",
  \"action\": \"on\"
}"""
            )

            btn_manual_intent = gr.Button("Use Manual Intent / Slots")

        with gr.Row():
            btn_intent = gr.Button("Run Intent Detection")

    with gr.Group():
        gr.Markdown("## Fulfillment")

        api_box = gr.Textbox(label="Fulfillment / API Output", lines=8)

        with gr.Accordion("Manual / Bypass Options for Fulfillment", open=False):
            manual_api_result = gr.Textbox(
                label="Manual API Result (JSON)",
                lines=6,
                value="""{
  \"status\": \"success\",
  \"message\": \"lamp turned on\"
}"""
            )

            btn_manual_api = gr.Button("Use Manual API Result")

        with gr.Row():
            btn_fulfill = gr.Button("Run Fulfillment")

    with gr.Group():
        gr.Markdown("## Answer Generation and TTS")

        answer_box = gr.Textbox(label="Final Answer", lines=3)
        tts_output = gr.Textbox(label="TTS Output", lines=2)

        with gr.Row():
            btn_answer = gr.Button("Generate Answer")
            btn_tts = gr.Button("Run TTS")
            btn_reset = gr.Button("Reset All")

    btn_verify.click(
        fn=do_verify,
        inputs=[audio_input, state],
        outputs=[verify_output, verification_scores_output, state, assistant_status]
    )

    btn_skip_verify.click(
        fn=verify_with_code,
        inputs=[verification_code_input, state],
        outputs=[verify_output, verification_scores_output, state, assistant_status]
    )

    btn_reset_verification.click(
        fn=reset_verification,
        inputs=[state],
        outputs=[verify_output, verification_code_input, verification_scores_output, state, assistant_status]
    )

    btn_wake.click(
        fn=do_wake,
        inputs=[audio_input, state],
        outputs=[wake_output, state, assistant_status]
    )

    btn_skip_wake.click(
        fn=skip_wake,
        inputs=[state],
        outputs=[wake_output, state, assistant_status]
    )

    btn_asr.click(
        fn=do_asr,
        inputs=[audio_input, state],
        outputs=[transcript_box, state, assistant_status]
    )

    btn_intent.click(
        fn=do_intent,
        inputs=[transcript_box, state],
        outputs=[intent_box, slots_box, state, assistant_status]
    )

    btn_manual_intent.click(
        fn=use_manual_intent,
        inputs=[manual_intent, manual_slots, state],
        outputs=[intent_box, slots_box, state, assistant_status]
    )

    btn_fulfill.click(
        fn=do_fulfillment,
        inputs=[state],
        outputs=[api_box, control_box, state, assistant_status]
    )

    btn_manual_api.click(
        fn=use_manual_api_result,
        inputs=[manual_api_result, state],
        outputs=[api_box, control_box, state, assistant_status]
    )

    btn_answer.click(
        fn=do_answer,
        inputs=[state],
        outputs=[answer_box, state, assistant_status]
    )

    btn_tts.click(
        fn=do_tts,
        inputs=[state],
        outputs=[tts_output]
    )

    btn_reset.click(
        fn=reset_all,
        inputs=[],
        outputs=[
            state,
            assistant_status,
            verify_output,
            verification_code_input,
            verification_scores_output,
            wake_output,
            transcript_box,
            intent_box,
            slots_box,
            api_box,
            control_box,
            answer_box,
            tts_output,
            manual_intent,
            manual_slots,
            manual_api_result
        ]
    )

demo.launch(ssr_mode=False)
