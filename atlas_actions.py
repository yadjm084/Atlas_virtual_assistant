"""Action handlers and UI wrapper callbacks for Atlas."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from pathlib import Path

from atlas_state import (
    clear_from_fulfillment,
    clear_from_transcript,
    get_pipeline_ui_updates,
    init_state as init_state_factory,
)


@dataclass
class AtlasRuntime:
    default_control_state: dict
    valid_codes: list[str]
    chosen_threshold: float
    wake_threshold: float
    ready_window_seconds: int
    default_tts_voice: str
    tts_output_dir: Path
    speaker_profiles: dict
    feature_scaler: object
    profile_load_status: str
    wake_model: object
    wake_model_status: str
    asr_model: object
    asr_model_status: str
    intent_predictor: object
    intent_model_status: str
    tts_status: str
    verify_audio_file: callable
    predict_wake_word: callable
    transcribe_with_whisper: callable
    fulfill_intent: callable
    generate_answer: callable
    fulfillment_error_cls: type[Exception]
    synthesize_tts_audio: callable


class AtlasActions:
    def __init__(self, runtime: AtlasRuntime):
        self.runtime = runtime

    def init_state(self):
        return init_state_factory(self.runtime.default_control_state)

    def set_ready_window(self, state):
        state["ready_until"] = time.time() + self.runtime.ready_window_seconds

    def clear_command_context(self, state):
        state["awake"] = False
        state["wake_completed"] = False
        state["ready_until"] = None
        state["wake_probability"] = None
        state["transcript"] = ""
        clear_from_transcript(state)

    def ensure_ready_state(self, state):
        if state["awake"] and state["ready_until"] is not None:
            if time.time() >= state["ready_until"]:
                self.clear_command_context(state)

    def get_ready_time_left(self, state):
        self.ensure_ready_state(state)
        if not state["awake"] or state["ready_until"] is None:
            return 0
        remaining = state["ready_until"] - time.time()
        return max(0, math.ceil(remaining))

    def get_ready_countdown_text(self, state):
        seconds_left = self.get_ready_time_left(state)
        if state["awake"] and seconds_left > 0:
            return f"{seconds_left} s before sleep"
        return "Sleeping"

    def get_status_text(self, state):
        self.ensure_ready_state(state)
        verify_score_text = f"{state['verification_best_score']:.4f}" if state["verification_best_score"] is not None else "None"
        wake_score_text = f"{state['wake_probability']:.4f}" if state["wake_probability"] is not None else "None"
        intent_score_text = f"{state['intent_confidence']:.4f}" if state["intent_confidence"] is not None else "None"
        ready_left = self.get_ready_time_left(state)
        return (
            f"Verified: {state['verified']}\n"
            f"Verified User: {state['verified_user']}\n"
            f"Best Verification Score: {verify_score_text}\n"
            f"Awake: {state['awake']}\n"
            f"Wake Probability: {wake_score_text}\n"
            f"Ready Time Left: {ready_left} s\n"
            f"Intent: {state['intent'] if state['intent'] else 'None'}\n"
            f"Intent Confidence: {intent_score_text}"
        )

    def _pipeline_updates(self, state):
        return get_pipeline_ui_updates(state, self.get_status_text(state), self.get_ready_time_left(state))

    def tick_ready_timer(self, state):
        self.ensure_ready_state(state)
        return state, self.get_status_text(state), self.get_ready_countdown_text(state)

    def do_verify(self, audio, state):
        if audio is None:
            return "Please record or upload an audio file first.", "{}", state, self.get_status_text(state)
        if not self.runtime.speaker_profiles or self.runtime.feature_scaler is None:
            return f"Speaker profiles could not be loaded. {self.runtime.profile_load_status}", "{}", state, self.get_status_text(state)

        try:
            result = self.runtime.verify_audio_file(
                audio_path=audio,
                speaker_profiles=self.runtime.speaker_profiles,
                scaler=self.runtime.feature_scaler,
                threshold=self.runtime.chosen_threshold,
            )
            state["verification_scores"] = result["all_scores"]
            state["verification_best_score"] = result["best_score"]

            if result["accepted"]:
                state["verified"] = True
                state["verified_user"] = result["predicted_user"]
                self.clear_command_context(state)
                message = (
                    f"Verification successful. User identified as {result['predicted_user']} "
                    f"(score={result['best_score']:.4f}, threshold={self.runtime.chosen_threshold})."
                )
            else:
                state["verified"] = False
                state["verified_user"] = "None"
                state["awake"] = False
                state["wake_completed"] = False
                state["ready_until"] = None
                state["wake_probability"] = None
                state["transcript"] = ""
                clear_from_transcript(state)
                message = (
                    f"Verification failed. Atlas remains locked "
                    f"(best score={result['best_score']:.4f}, threshold={self.runtime.chosen_threshold})."
                )

            return message, json.dumps(result["all_scores"], indent=2), state, self.get_status_text(state)
        except Exception as e:
            return f"Verification error: {e}", "{}", state, self.get_status_text(state)

    def verify_with_code(self, code_input, state):
        code = code_input.strip()
        if code in self.runtime.valid_codes:
            state["verified"] = True
            state["verified_user"] = code
            state["verification_best_score"] = None
            state["verification_scores"] = {}
            self.clear_command_context(state)
            return f"Verification bypass successful. User set to {code}.", "{}", state, self.get_status_text(state)
        return "Invalid verification code. Use Adjmal, Nair, or Sharma.", "{}", state, self.get_status_text(state)

    def reset_verification(self, state):
        state["verified"] = False
        state["verified_user"] = "None"
        state["verification_scores"] = {}
        state["verification_best_score"] = None
        self.clear_command_context(state)
        return "", "", "{}", state, self.get_status_text(state)

    def do_wake(self, audio, state):
        if not state["verified"]:
            return "Please complete user verification first.", state, self.get_status_text(state)
        if audio is None:
            return "Please record or upload an audio file first.", state, self.get_status_text(state)
        if self.runtime.wake_model is None:
            return f"Wake word model unavailable. {self.runtime.wake_model_status}", state, self.get_status_text(state)

        try:
            result = self.runtime.predict_wake_word(
                audio_path=audio,
                model=self.runtime.wake_model,
                target_sr=16000,
                target_length=32000,
                n_mfcc=13,
                threshold=self.runtime.wake_threshold,
            )
            state["wake_probability"] = result["probability_positive"]
            if result["wake_detected"]:
                state["awake"] = True
                state["wake_completed"] = True
                self.set_ready_window(state)
                message = (
                    f"Wake word detected. Atlas is now awake "
                    f"(probability={result['probability_positive']:.4f}, threshold={self.runtime.wake_threshold}, "
                    f"ready_window={self.runtime.ready_window_seconds}s)."
                )
            else:
                self.clear_command_context(state)
                message = (
                    f"Wake word not detected "
                    f"(probability={result['probability_positive']:.4f}, threshold={self.runtime.wake_threshold})."
                )
            return message, state, self.get_status_text(state)
        except Exception as e:
            return f"Wake word error: {e}", state, self.get_status_text(state)

    def skip_wake_with_code(self, wake_code_input, state):
        if not state["verified"]:
            return "Please complete user verification first.", state, self.get_status_text(state)
        code = wake_code_input.strip()
        if code == "Hey Atlas":
            state["awake"] = True
            state["wake_completed"] = True
            state["wake_probability"] = None
            state["transcript"] = ""
            clear_from_transcript(state)
            self.set_ready_window(state)
            return (
                f"Wake word bypass successful. Atlas is now awake for {self.runtime.ready_window_seconds} seconds.",
                state,
                self.get_status_text(state),
            )
        return "Invalid wake word bypass code. Use exactly: Hey Atlas", state, self.get_status_text(state)

    def reset_wake_word(self, state):
        self.clear_command_context(state)
        return "", "", state, self.get_status_text(state)

    def do_asr(self, audio, state):
        self.ensure_ready_state(state)
        if not state["verified"]:
            return "Please complete user verification first.", state, self.get_status_text(state)
        if not state["awake"]:
            return "Please detect the wake word first.", state, self.get_status_text(state)
        if audio is None:
            return "Please record or upload a command audio file first.", state, self.get_status_text(state)
        if self.runtime.asr_model is None:
            return f"ASR model unavailable. {self.runtime.asr_model_status}", state, self.get_status_text(state)
        try:
            transcript = self.runtime.transcribe_with_whisper(audio, self.runtime.asr_model)
            clear_from_transcript(state)
            state["transcript"] = transcript
            self.set_ready_window(state)
            return transcript, state, self.get_status_text(state)
        except Exception as e:
            return f"ASR error: {e}", state, self.get_status_text(state)

    def use_typed_transcript(self, typed_text, state):
        self.ensure_ready_state(state)
        if not state["verified"]:
            return "Please complete user verification first.", state, self.get_status_text(state)
        if not state["awake"]:
            return "Please detect the wake word first.", state, self.get_status_text(state)
        typed_text = typed_text.strip()
        if not typed_text:
            return "Please type a sentence first.", state, self.get_status_text(state)
        clear_from_transcript(state)
        state["transcript"] = typed_text
        self.set_ready_window(state)
        return typed_text, state, self.get_status_text(state)

    def do_intent(self, transcript, state):
        self.ensure_ready_state(state)
        if not state["verified"]:
            return "Verification required.", "{}", state, self.get_status_text(state)
        if not state["awake"]:
            return "Wake word required.", "{}", state, self.get_status_text(state)
        if not transcript.strip():
            return "No transcript available.", "{}", state, self.get_status_text(state)
        if self.runtime.intent_predictor is None:
            return self.runtime.intent_model_status, "{}", state, self.get_status_text(state)

        try:
            prediction = self.runtime.intent_predictor.predict(transcript)
        except Exception as e:
            return f"Intent detection failed: {e}", "{}", state, self.get_status_text(state)

        clear_from_fulfillment(state)
        state["intent"] = prediction.intent
        state["intent_confidence"] = prediction.intent_confidence
        state["slots"] = prediction.slots
        self.set_ready_window(state)
        return state["intent"], json.dumps(state["slots"], indent=2), state, self.get_status_text(state)

    def use_manual_intent(self, manual_intent, manual_slots, state):
        clear_from_fulfillment(state)
        state["intent"] = manual_intent.strip()
        state["intent_confidence"] = None
        try:
            state["slots"] = json.loads(manual_slots) if manual_slots.strip() else {}
            return state["intent"], json.dumps(state["slots"], indent=2), state, self.get_status_text(state)
        except json.JSONDecodeError:
            return state["intent"], "Invalid JSON format in manual slots.", state, self.get_status_text(state)

    def do_fulfillment(self, state):
        self.ensure_ready_state(state)
        if not state["intent"]:
            return "Please detect or enter an intent first.", json.dumps(state["control_state"], indent=2), state, self.get_status_text(state)
        if not state["awake"]:
            return "Atlas went to sleep. Please say the wake word again.", json.dumps(state["control_state"], indent=2), state, self.get_status_text(state)

        try:
            api_result, updated_control_state = self.runtime.fulfill_intent(
                state["intent"], state["slots"], state["control_state"]
            )
            state["control_state"] = updated_control_state
        except self.runtime.fulfillment_error_cls as e:
            api_result = {"status": "error", "message": str(e)}
        except Exception as e:
            api_result = {"status": "error", "message": f"Fulfillment failed: {e}"}

        state["api_result"] = api_result
        state["answer_text"] = ""
        state["last_tts_path"] = None
        self.set_ready_window(state)
        return json.dumps(api_result, indent=2), json.dumps(state["control_state"], indent=2), state, self.get_status_text(state)

    def use_manual_api_result(self, manual_api_result, state):
        try:
            state["api_result"] = json.loads(manual_api_result) if manual_api_result.strip() else {}
            state["answer_text"] = ""
            state["last_tts_path"] = None
            return json.dumps(state["api_result"], indent=2), json.dumps(state["control_state"], indent=2), state, self.get_status_text(state)
        except json.JSONDecodeError:
            return "Invalid JSON format in manual API result.", json.dumps(state["control_state"], indent=2), state, self.get_status_text(state)

    def do_answer(self, state):
        self.ensure_ready_state(state)
        answer = self.runtime.generate_answer(
            intent=state["intent"],
            slots=state["slots"],
            api_result=state["api_result"],
            control_state=state["control_state"],
        )
        state["answer_text"] = answer
        return answer, state, self.get_status_text(state)

    def use_manual_answer(self, manual_answer, state):
        manual_answer = manual_answer.strip()
        if not manual_answer:
            return "Please enter a manual answer first.", state, self.get_status_text(state)
        state["answer_text"] = manual_answer
        return manual_answer, state, self.get_status_text(state)

    def do_tts(self, state):
        answer_text = state["answer_text"].strip()
        if not answer_text:
            if state["api_result"]:
                answer_text = state["api_result"].get("message", "").strip()
            if not answer_text:
                return "Generate an answer first.", None, state
        try:
            audio_path = self.runtime.synthesize_tts_audio(answer_text, self.runtime.tts_output_dir)
            state["last_tts_path"] = str(audio_path)
            return f"TTS generated with {self.runtime.default_tts_voice}.", str(audio_path), state
        except Exception as e:
            return f"TTS failed: {e}", None, state

    def reset_all(self):
        state = self.init_state()
        return (
            state,
            None,
            None,
            None,
            self.get_status_text(state),
            "Sleeping",
            "",
            "{}",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            json.dumps(state["control_state"], indent=2),
            "",
            "",
            None,
            "The bedroom light is now on.",
            "LightOn",
            '''{
  "ROOM": "bedroom"
}''',
            '''{
  "status": "success",
  "message": "bedroom light turned on"
}''',
        )

    # UI wrapper callbacks
    def do_verify_ui(self, audio, state):
        message, scores, state, _ = self.do_verify(audio, state)
        return message, scores, state, *self._pipeline_updates(state)

    def verify_with_code_ui(self, code_input, state):
        message, scores, state, _ = self.verify_with_code(code_input, state)
        return message, scores, state, *self._pipeline_updates(state)

    def reset_verification_ui(self, state):
        verify_output, verification_code, scores, state, _ = self.reset_verification(state)
        return verify_output, verification_code, scores, state, *self._pipeline_updates(state)

    def do_wake_ui(self, audio, state):
        wake_output, state, _ = self.do_wake(audio, state)
        return wake_output, state, *self._pipeline_updates(state)

    def skip_wake_with_code_ui(self, wake_code_input, state):
        wake_output, state, _ = self.skip_wake_with_code(wake_code_input, state)
        return wake_output, state, *self._pipeline_updates(state)

    def reset_wake_word_ui(self, state):
        wake_output, wake_code, state, _ = self.reset_wake_word(state)
        return wake_output, wake_code, state, *self._pipeline_updates(state)

    def do_asr_ui(self, audio, state):
        transcript, state, _ = self.do_asr(audio, state)
        return transcript, state, *self._pipeline_updates(state)

    def use_typed_transcript_ui(self, typed_text, state):
        transcript, state, _ = self.use_typed_transcript(typed_text, state)
        return transcript, state, *self._pipeline_updates(state)

    def do_intent_ui(self, transcript, state):
        intent, slots, state, _ = self.do_intent(transcript, state)
        return intent, slots, state, *self._pipeline_updates(state)

    def use_manual_intent_ui(self, manual_intent, manual_slots, state):
        intent, slots, state, _ = self.use_manual_intent(manual_intent, manual_slots, state)
        return intent, slots, state, *self._pipeline_updates(state)

    def do_fulfillment_ui(self, state):
        api_output, control_output, state, _ = self.do_fulfillment(state)
        return api_output, control_output, state, *self._pipeline_updates(state)

    def use_manual_api_result_ui(self, manual_api_result, state):
        api_output, control_output, state, _ = self.use_manual_api_result(manual_api_result, state)
        return api_output, control_output, state, *self._pipeline_updates(state)

    def do_answer_ui(self, state):
        answer, state, _ = self.do_answer(state)
        return answer, state, *self._pipeline_updates(state)

    def use_manual_answer_ui(self, manual_answer, state):
        answer, state, _ = self.use_manual_answer(manual_answer, state)
        return answer, state, *self._pipeline_updates(state)

    def do_tts_ui(self, state):
        tts_status, tts_audio, state = self.do_tts(state)
        return tts_status, tts_audio, state, *self._pipeline_updates(state)

    def reset_all_ui(self):
        outputs = self.reset_all()
        state = outputs[0]
        return (*outputs, *self._pipeline_updates(state))

    def tick_ready_timer_ui(self, state):
        state, raw_status, countdown = self.tick_ready_timer(state)
        pipeline_updates = self._pipeline_updates(state)
        return state, *pipeline_updates[:2], countdown, *pipeline_updates[2:]
