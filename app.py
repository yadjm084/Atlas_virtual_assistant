import gradio as gr
import json
import os
from pathlib import Path

import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# USER VERIFICATION CONFIGURATION
# =========================================================

# Folder containing the enrollment/test audio data.
# This assumes that "user_verification" is located in the same folder as app.py.
BASE_DIR = Path(__file__).resolve().parent
USER_VERIFICATION_DIR = BASE_DIR / "user_verification"
ENROLLMENT_DIR = USER_VERIFICATION_DIR / "enrollment"

print("BASE_DIR =", BASE_DIR)
print("USER_VERIFICATION_DIR exists =", USER_VERIFICATION_DIR.exists(), USER_VERIFICATION_DIR)
print("ENROLLMENT_DIR exists =", ENROLLMENT_DIR.exists(), ENROLLMENT_DIR)

if ENROLLMENT_DIR.exists():
    print("Enrollment subfolders:")
    for item in ENROLLMENT_DIR.iterdir():
        print("-", item, "DIR" if item.is_dir() else "FILE")
# Audio preprocessing parameters chosen from your notebook
TARGET_SR = 16000
TARGET_DURATION = 2.5
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)
N_MFCC = 13

# Security-oriented threshold selected from your experiments
CHOSEN_THRESHOLD = 0.92

# List of valid bypass codes / enrolled users
VALID_CODES = ["Adjmal", "Nair", "Sharma"]


# =========================================================
# USER VERIFICATION HELPER FUNCTIONS
# =========================================================

def load_and_preprocess_audio(file_path, target_sr=16000, target_length=40000):
    """
    Load an audio file, resample it, and force it to a fixed length.

    Steps:
    1. Load the audio file with librosa
    2. Resample to the target sampling rate
    3. If the signal is longer than target_length, truncate it
    4. If the signal is shorter than target_length, pad it with zeros

    Returns:
        signal (np.array): preprocessed fixed-length audio
        sr (int): sampling rate used
    """
    signal, sr = librosa.load(file_path, sr=target_sr)

    if len(signal) > target_length:
        signal = signal[:target_length]
    elif len(signal) < target_length:
        pad_amount = target_length - len(signal)
        signal = np.pad(signal, (0, pad_amount), mode="constant")

    return signal, sr


def extract_mfcc(signal, sr, n_mfcc=13):
    """
    Extract MFCC features from a preprocessed audio signal.

    Returns:
        mfcc (np.array): shape = (n_mfcc, time_frames)
    """
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc
    )
    return mfcc


def mfcc_to_fixed_vector(mfcc):
    """
    Convert an MFCC matrix into a fixed-size speaker vector using:
    - mean of each coefficient over time
    - standard deviation of each coefficient over time

    Returns:
        feature_vector (np.array): shape = (2 * n_mfcc,)
    """
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)
    feature_vector = np.concatenate([mfcc_means, mfcc_stds])
    return feature_vector


def extract_speaker_vector(file_path, target_sr=16000, target_length=40000, n_mfcc=13):
    """
    Full pipeline:
    1. Load and preprocess the audio
    2. Extract MFCCs
    3. Convert MFCCs to a fixed-size vector
    """
    signal, sr = load_and_preprocess_audio(
        file_path=file_path,
        target_sr=target_sr,
        target_length=target_length
    )
    mfcc = extract_mfcc(signal, sr, n_mfcc=n_mfcc)
    feature_vector = mfcc_to_fixed_vector(mfcc)
    return feature_vector


def load_enrollment_profiles(enrollment_dir):
    """
    Build one average voice profile per enrolled speaker.

    Returns:
        speaker_profiles (dict): {speaker_name: mean_profile_vector}
        status_message (str): summary of loading result
    """
    speaker_profiles = {}
    messages = []

    print("DEBUG - enrollment_dir:", enrollment_dir)
    print("DEBUG - enrollment_dir exists:", enrollment_dir.exists())

    if not enrollment_dir.exists():
        return {}, f"Enrollment directory not found: {enrollment_dir}"

    for speaker_folder in enrollment_dir.iterdir():
        if not speaker_folder.is_dir():
            continue

        speaker = speaker_folder.name
        audio_files = list(speaker_folder.glob("*.wav"))

        print(f"DEBUG - speaker folder: {speaker_folder}")
        print(f"DEBUG - found {len(audio_files)} .m4a files for {speaker}")

        if len(audio_files) == 0:
            messages.append(f"{speaker}: no enrollment files found")
            continue

        speaker_vectors = []

        for audio_file in audio_files:
            try:
                print(f"DEBUG - processing {audio_file}")
                vector = extract_speaker_vector(
                    file_path=str(audio_file),
                    target_sr=TARGET_SR,
                    target_length=TARGET_LENGTH,
                    n_mfcc=N_MFCC
                )
                speaker_vectors.append(vector)
            except Exception as e:
                print(f"DEBUG - failed on {audio_file}: {e}")
                messages.append(f"{speaker}: failed on {audio_file.name} ({e})")

        if len(speaker_vectors) > 0:
            mean_profile = np.mean(np.vstack(speaker_vectors), axis=0)
            speaker_profiles[speaker] = mean_profile
            messages.append(f"{speaker}: loaded {len(speaker_vectors)} enrollment files")

    if not speaker_profiles:
        return {}, "No speaker profiles could be built. " + " | ".join(messages)

    return speaker_profiles, " | ".join(messages)

def compare_to_profiles(test_vector, speaker_profiles):
    """
    Compare one test vector to all enrolled speaker profiles
    using cosine similarity.

    Returns:
        scores (dict): {speaker_name: cosine_similarity_score}
    """
    scores = {}
    test_vector_2d = test_vector.reshape(1, -1)

    for speaker, profile_vector in speaker_profiles.items():
        profile_vector_2d = profile_vector.reshape(1, -1)
        score = cosine_similarity(test_vector_2d, profile_vector_2d)[0][0]
        scores[speaker] = float(score)

    return scores


def verify_audio_file(audio_path, speaker_profiles, threshold=0.995):
    """
    Verify a single input audio file against enrolled speaker profiles.

    Returns:
        result (dict) with:
            - accepted (bool)
            - predicted_user (str)
            - best_score (float)
            - all_scores (dict)
    """
    test_vector = extract_speaker_vector(
        file_path=audio_path,
        target_sr=TARGET_SR,
        target_length=TARGET_LENGTH,
        n_mfcc=N_MFCC
    )

    scores = compare_to_profiles(test_vector, speaker_profiles)
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


# Load speaker profiles once at startup
SPEAKER_PROFILES, PROFILE_LOAD_STATUS = load_enrollment_profiles(ENROLLMENT_DIR)


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
        "control_state": {
            "lamp": "off",
            "temperature": 20
        }
    }


def get_status_text(state):
    score_text = (
        f"{state['verification_best_score']:.4f}"
        if state["verification_best_score"] is not None
        else "None"
    )

    return (
        f"Verified: {state['verified']}\n"
        f"Verified User: {state['verified_user']}\n"
        f"Best Verification Score: {score_text}\n"
        f"Awake: {state['awake']}\n"
        f"Intent: {state['intent'] if state['intent'] else 'None'}"
    )


# =========================================================
# USER VERIFICATION FUNCTIONS
# =========================================================

def do_verify(audio, state):
    """
    Run real speaker verification on the provided audio file.
    """
    if audio is None:
        return "Please record or upload an audio file first.", "{}", state, get_status_text(state)

    if not SPEAKER_PROFILES:
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

            message = (
                f"Verification failed. No enrolled user passed the threshold "
                f"(best match score={result['best_score']:.4f}, threshold={CHOSEN_THRESHOLD})."
            )

        scores_json = json.dumps(result["all_scores"], indent=2)
        return message, scores_json, state, get_status_text(state)

    except Exception as e:
        return f"Verification error: {e}", "{}", state, get_status_text(state)


def verify_with_code(code_input, state):
    """
    Verification bypass using a code.
    Here, the accepted codes are the enrolled user names.
    """
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
    else:
        return (
            "Invalid verification code. Use Adjmal, Nair, or Sharma.",
            "{}",
            state,
            get_status_text(state)
        )


def reset_verification(state):
    """
    Reset only verification-related values.
    """
    state["verified"] = False
    state["verified_user"] = "None"
    state["verification_scores"] = {}
    state["verification_best_score"] = None

    return "", "", "{}", state, get_status_text(state)


# =========================================================
# WAKE WORD FUNCTIONS
# =========================================================

def do_wake(audio, state):
    if not state["verified"]:
        return "Please complete user verification first.", state, get_status_text(state)

    state["awake"] = True
    return "Wake word detected (placeholder).", state, get_status_text(state)


def skip_wake(state):
    if not state["verified"]:
        return "Please complete user verification first.", state, get_status_text(state)

    state["awake"] = True
    return "Wake word detection skipped.", state, get_status_text(state)


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
        get_status_text(state),  # assistant status
        "",   # verification output
        "",   # verification code input
        "{}", # verification scores output
        "",   # wake word output
        "",   # transcript
        "",   # intent
        "",   # slots
        "",   # api result
        json.dumps(state["control_state"], indent=2),  # control state
        "",   # final answer
        "",   # tts output
        "control_device",  # manual intent
        '{\n  "device": "lamp",\n  "action": "on"\n}',  # manual slots
        '{\n  "status": "success",\n  "message": "lamp turned on"\n}'  # manual api result
    )


# =========================================================
# GRADIO INTERFACE
# =========================================================

with gr.Blocks(title="Atlas - Virtual Assistant") as demo:
    gr.Markdown("# Atlas - Virtual Assistant")
    gr.Markdown("User verification now uses real enrollment profiles and cosine-similarity matching.")

    state = gr.State(init_state())

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Voice Input")

        assistant_status = gr.Textbox(
            label="Assistant Status",
            value=get_status_text(init_state()),
            lines=5
        )

        control_box = gr.Textbox(
            label="Control System State",
            value=json.dumps(init_state()["control_state"], indent=2),
            lines=8
        )

    with gr.Group():
        gr.Markdown("## User Verification")
        gr.Markdown(f"Enrollment status: {PROFILE_LOAD_STATUS}")
        gr.Markdown(f"Threshold: {CHOSEN_THRESHOLD}")

        verify_output = gr.Textbox(
            label="Verification Output",
            lines=3
        )

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

        wake_output = gr.Textbox(label="Wake Word Output", lines=2)

        with gr.Row():
            btn_wake = gr.Button("Run Wake Word")
            btn_skip_wake = gr.Button("Skip Wake Word")

    with gr.Group():
        gr.Markdown("## Speech Recognition")
        transcript_box = gr.Textbox(label="Transcript", lines=3)

        with gr.Row():
            btn_asr = gr.Button("Run ASR")

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
                value='{\n  "device": "lamp",\n  "action": "on"\n}'
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
                value='{\n  "status": "success",\n  "message": "lamp turned on"\n}'
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

    # =====================================================
    # BUTTON CONNECTIONS
    # =====================================================

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

demo.launch()