from pathlib import Path
from functools import partial

from atlas_actions import AtlasActions, AtlasRuntime
from atlas_answer_generation import generate_answer
from atlas_fulfillment import DEFAULT_CONTROL_STATE, FulfillmentError, fulfill_intent
from atlas_tts import DEFAULT_TTS_VOICE, EDGE_TTS_STATUS, synthesize_tts_audio
from atlas_ui import ATLAS_CSS, ATLAS_THEME, build_demo
from atlas_voice import (
    get_dataset_root,
    get_enrollment_dir,
    get_wake_weights_path,
    load_asr_model,
    load_enrollment_profiles_with_normalization,
    load_wake_model,
    predict_wake_word,
    transcribe_with_whisper,
    verify_audio_file,
)
from intent_data.intent_inference import load_intent_predictor


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
HF_DATASET_REPO = "yadjm084/atlas-voice-data"
HF_INTENT_REPO = "yadjm084/atlas-intent-data"

TARGET_SR = 16000
TARGET_DURATION = 2.5
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)
N_MFCC = 13

CHOSEN_THRESHOLD = 0.5
VALID_CODES = ["Adjmal", "Nair", "Sharma"]

WAKE_TARGET_SR = 16000
WAKE_TARGET_DURATION = 2.0
WAKE_TARGET_LENGTH = int(WAKE_TARGET_SR * WAKE_TARGET_DURATION)
WAKE_N_MFCC = 13
WAKE_THRESHOLD = 0.5

ASR_MODEL_NAME = "tiny"
READY_WINDOW_SECONDS = 20

TTS_OUTPUT_DIR = BASE_DIR / "generated_audio"


# =========================================================
# MODEL / DATASET BOOTSTRAP
# =========================================================

# Voice dataset
DATASET_ROOT = get_dataset_root(BASE_DIR, HF_DATASET_REPO)
ENROLLMENT_DIR = get_enrollment_dir(DATASET_ROOT)
WAKE_WEIGHTS_PATH = get_wake_weights_path(DATASET_ROOT)

# Intent dataset
INTENT_DATASET_ROOT = get_dataset_root(BASE_DIR, HF_INTENT_REPO)
INTENT_ARTIFACTS_DIR = INTENT_DATASET_ROOT

SPEAKER_PROFILES, FEATURE_SCALER, PROFILE_LOAD_STATUS = load_enrollment_profiles_with_normalization(
    ENROLLMENT_DIR,
    target_sr=TARGET_SR,
    target_length=TARGET_LENGTH,
    n_mfcc=N_MFCC,
)

WAKE_MODEL, WAKE_MODEL_STATUS = load_wake_model(WAKE_WEIGHTS_PATH)
ASR_MODEL, ASR_MODEL_STATUS = load_asr_model(ASR_MODEL_NAME)
INTENT_PREDICTOR, INTENT_MODEL_STATUS = load_intent_predictor(INTENT_ARTIFACTS_DIR)


# =========================================================
# RUNTIME WIRING
# =========================================================

runtime = AtlasRuntime(
    default_control_state=DEFAULT_CONTROL_STATE,
    valid_codes=VALID_CODES,
    chosen_threshold=CHOSEN_THRESHOLD,
    wake_threshold=WAKE_THRESHOLD,
    ready_window_seconds=READY_WINDOW_SECONDS,
    default_tts_voice=DEFAULT_TTS_VOICE,
    tts_output_dir=TTS_OUTPUT_DIR,
    speaker_profiles=SPEAKER_PROFILES,
    feature_scaler=FEATURE_SCALER,
    profile_load_status=PROFILE_LOAD_STATUS,
    wake_model=WAKE_MODEL,
    wake_model_status=WAKE_MODEL_STATUS,
    asr_model=ASR_MODEL,
    asr_model_status=ASR_MODEL_STATUS,
    intent_predictor=INTENT_PREDICTOR,
    intent_model_status=INTENT_MODEL_STATUS,
    tts_status=EDGE_TTS_STATUS,
    verify_audio_file=partial(
        verify_audio_file,
        target_sr=TARGET_SR,
        target_length=TARGET_LENGTH,
        n_mfcc=N_MFCC,
    ),
    predict_wake_word=partial(
        predict_wake_word,
        target_sr=WAKE_TARGET_SR,
        target_length=WAKE_TARGET_LENGTH,
        n_mfcc=WAKE_N_MFCC,
    ),
    transcribe_with_whisper=transcribe_with_whisper,
    fulfill_intent=fulfill_intent,
    generate_answer=generate_answer,
    fulfillment_error_cls=FulfillmentError,
    synthesize_tts_audio=synthesize_tts_audio,
)

actions = AtlasActions(runtime)


# Expose core functions for scripts/tests that import app.py directly.
init_state = actions.init_state
verify_with_code = actions.verify_with_code
skip_wake_with_code = actions.skip_wake_with_code
use_typed_transcript = actions.use_typed_transcript
do_intent = actions.do_intent
do_fulfillment = actions.do_fulfillment
do_answer = actions.do_answer
do_tts = actions.do_tts

demo = build_demo(actions)


if __name__ == "__main__":
    demo.launch(ssr_mode=False, theme=ATLAS_THEME, css=ATLAS_CSS)