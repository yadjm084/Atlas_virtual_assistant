"""Run repeatable end-to-end smoke tests for the Atlas demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import app


DEMO_FLOWS = [
    {
        "name": "weather",
        "transcript": "does it rain in Ottawa today",
        "expected_intent": "GetWeather",
        "required_slots": {"CITY": "Ottawa"},
    },
    {
        "name": "movie_director",
        "transcript": "who directed Dune Part Two",
        "expected_intent": "MovieDirector",
        "required_slots": {"TITLE": "Dune Part Two"},
    },
    {
        "name": "movie_similar",
        "transcript": "find movies like The Matrix",
        "expected_intent": "SimilarMovies",
        "required_slots": {"TITLE": "The Matrix"},
    },
    {
        "name": "light_on",
        "transcript": "turn on the theater light",
        "expected_intent": "LightOn",
        "required_slots": {"ROOM": "theater"},
    },
    {
        "name": "brightness",
        "transcript": "set the theater light to 40 percent",
        "expected_intent": "SetBrightness",
        "required_slots": {"ROOM": "theater", "BRIGHTNESS": "40 percent"},
    },
    {
        "name": "scene",
        "transcript": "switch to study mode",
        "expected_intent": "SetScene",
        "required_slots": {"SCENE": "study"},
    },
    {
        "name": "oos",
        "transcript": "book a flight to Toronto",
        "expected_intent": "OOS",
        "required_slots": {},
    },
]

VERIFICATION_AUDIO_SAMPLES = [
    ("Adjmal", "Adjmal/Adjmal-extra-1.wav"),
    ("Nair", "Nair/Nair-positive-1.wav"),
    ("Sharma", "Sharma/Sharma-positive-2.wav"),
]

WAKE_AUDIO_SAMPLES = [
    (True, "Adjmal/Adjmal-positive-1.wav"),
    (False, "Adjmal/Adjmal-other-1.wav"),
    (False, "Adjmal/Adjmal-near-1.wav"),
]


def _find_existing_sample(base_dir: Path, candidates: list[str]) -> Path:
    for relative in candidates:
        path = base_dir / relative
        if path.exists():
            return path
    raise AssertionError(f"None of the candidate sample files exist under {base_dir}: {candidates}")

def assert_slot_subset(actual: dict, expected: dict):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise AssertionError(f"Expected slot {key}={value!r}, got {actual.get(key)!r}")


def run_flow(flow: dict, run_tts: bool = True):
    state = app.init_state()
    verify_msg, _, state, _ = app.verify_with_code("Nair", state)
    wake_msg, state, _ = app.skip_wake_with_code("Hey Atlas", state)
    _, state, _ = app.use_typed_transcript(flow["transcript"], state)
    intent, slots_json, state, _ = app.do_intent(flow["transcript"], state)
    slots = json.loads(slots_json)
    api_json, control_json, state, _ = app.do_fulfillment(state)
    api = json.loads(api_json)
    control = json.loads(control_json)
    answer, state, _ = app.do_answer(state)

    tts_status = None
    tts_audio = None
    if run_tts:
        tts_status, tts_audio, state = app.do_tts(state)

    if intent != flow["expected_intent"]:
        raise AssertionError(f"{flow['name']}: expected intent {flow['expected_intent']}, got {intent}")
    assert_slot_subset(slots, flow["required_slots"])

    if api.get("status") != "success":
        raise AssertionError(f"{flow['name']}: expected success API result, got {api}")

    if not answer.strip():
        raise AssertionError(f"{flow['name']}: answer text is empty")

    if run_tts:
        if not tts_audio:
            raise AssertionError(f"{flow['name']}: TTS did not return an audio path")
        audio_path = Path(tts_audio)
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            raise AssertionError(f"{flow['name']}: TTS output missing or empty at {tts_audio}")

    return {
        "verify": verify_msg,
        "wake": wake_msg,
        "intent": intent,
        "slots": slots,
        "api": api,
        "control": control,
        "answer": answer,
        "tts_status": tts_status,
        "tts_audio": tts_audio,
    }


def run_audio_gate_checks():
    verification_candidates = {
        "Adjmal": [
            "Adjmal/Adjmal-extra-1.wav",
            "Adjmal/Adjmal-extra-1.m4a",
        ],
        "Nair": [
            "Nair/Nair-positive-1.wav",
            "Nair/Nair-positive-1.m4a",
        ],
        "Sharma": [
            "Sharma/Sharma-positive-2.wav",
            "Sharma/Sharma-positive-2.m4a",
        ],
    }

    for expected_user, candidates in verification_candidates.items():
        sample_path = _find_existing_sample(app.ENROLLMENT_DIR, candidates)
        result = app.runtime.verify_audio_file(
            audio_path=str(sample_path),
            speaker_profiles=app.SPEAKER_PROFILES,
            scaler=app.FEATURE_SCALER,
            threshold=app.CHOSEN_THRESHOLD,
        )
        if result["predicted_user"] != expected_user or not result["accepted"]:
            raise AssertionError(
                f"Verification failed for {sample_path}: expected {expected_user}, got {result}"
            )
        print(
            f"[PASS] verify:{expected_user} -> {result['predicted_user']} "
            f"(score={result['best_score']:.4f}, file={sample_path.name})"
        )

    wake_candidates = [
        (
            True,
            [
                "Adjmal/Adjmal-positive-1.wav",
                "Adjmal/Adjmal-positive-1.m4a",
            ],
        ),
        (
            False,
            [
                "Adjmal/Adjmal-other-1.wav",
                "Adjmal/Adjmal-other-1.m4a",
            ],
        ),
        (
            False,
            [
                "Adjmal/Adjmal-near-1.wav",
                "Adjmal/Adjmal-near-1.m4a",
            ],
        ),
    ]

    for expected_detection, candidates in wake_candidates:
        sample_path = _find_existing_sample(app.ENROLLMENT_DIR, candidates)
        result = app.runtime.predict_wake_word(
            audio_path=str(sample_path),
            model=app.WAKE_MODEL,
            threshold=app.WAKE_THRESHOLD,
        )
        if result["wake_detected"] != expected_detection:
            raise AssertionError(
                f"Wake-word check failed for {sample_path}: expected {expected_detection}, got {result}"
            )
        print(
            f"[PASS] wake:{sample_path.name} -> detected={result['wake_detected']} "
            f"(p={result['probability_positive']:.4f})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tts", action="store_true", help="Skip audio generation to run faster.")
    parser.add_argument(
        "--audio-gates",
        action="store_true",
        help="Run real-audio verification and wake-word checks in addition to the bypass-based flow tests.",
    )
    args = parser.parse_args()

    print("Running Atlas demo smoke tests...")
    print(f"Intent model: {app.INTENT_MODEL_STATUS}")
    print(f"ASR model: {app.ASR_MODEL_STATUS}")
    print(f"Wake model: {app.WAKE_MODEL_STATUS}")

    for flow in DEMO_FLOWS:
        result = run_flow(flow, run_tts=not args.skip_tts)
        print(f"[PASS] {flow['name']}: {result['intent']} -> {result['answer']}")

    if args.audio_gates:
        run_audio_gate_checks()

    print("All demo smoke tests passed.")


if __name__ == "__main__":
    main()
