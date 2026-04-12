"""State, stage, and UI-summary helpers for Atlas."""

from __future__ import annotations

from copy import deepcopy
import gradio as gr


PIPELINE_STEPS = [
    ("identify", "User Identification"),
    ("wake", "Wake Word"),
    ("asr", "Speech to Text"),
    ("intent", "Intent Detection"),
    ("fulfill", "Action"),
    ("respond", "Answer + TTS"),
]


def init_state(default_control_state: dict) -> dict:
    return {
        "verified": False,
        "verified_user": "None",
        "awake": False,
        "ready_until": None,
        "transcript": "",
        "intent": "",
        "intent_confidence": None,
        "slots": {},
        "api_result": {},
        "answer_text": "",
        "last_tts_path": None,
        "verification_scores": {},
        "verification_best_score": None,
        "wake_probability": None,
        "wake_completed": False,
        "control_state": deepcopy(default_control_state),
    }


def clear_from_transcript(state: dict) -> None:
    state["intent"] = ""
    state["intent_confidence"] = None
    state["slots"] = {}
    state["api_result"] = {}
    state["answer_text"] = ""
    state["last_tts_path"] = None


def clear_from_fulfillment(state: dict) -> None:
    state["api_result"] = {}
    state["answer_text"] = ""
    state["last_tts_path"] = None


def get_current_stage_index(state: dict) -> int:
    if not state["verified"]:
        return 0
    if not state["wake_completed"]:
        return 1
    if not state["transcript"]:
        return 2
    if not state["intent"]:
        return 3
    if not state["api_result"]:
        return 4
    return 5


def build_pipeline_timeline_html(state: dict) -> str:
    current_index = get_current_stage_index(state)
    items = []

    for idx, (_, label) in enumerate(PIPELINE_STEPS):
        if idx < current_index:
            stage_class = "completed"
            stage_meta = "Done"
        elif idx == current_index:
            stage_class = "active"
            stage_meta = "Current"
        else:
            stage_class = "locked"
            stage_meta = "Locked"

        items.append(
            f"""
            <div class="atlas-stage {stage_class}">
              <div class="atlas-stage-index">{idx + 1}</div>
              <div class="atlas-stage-copy">
                <div class="atlas-stage-title">{label}</div>
                <div class="atlas-stage-meta">{stage_meta}</div>
              </div>
            </div>
            """
        )

    return f'<section class="atlas-timeline">{"".join(items)}</section>'


def build_status_overview_html(state: dict, ready_left: int) -> str:
    verified_badge = "Verified" if state["verified"] else "Locked"
    wake_badge = "Awake" if state["awake"] else "Sleeping"
    intent_label = state["intent"] if state["intent"] else "Pending"
    intent_conf = f"{state['intent_confidence']:.2f}" if state["intent_confidence"] is not None else "N/A"
    user_label = state["verified_user"] if state["verified"] else "None"

    return f"""
    <section class="atlas-status-summary">
      <div class="atlas-status-badges">
        <span class="atlas-status-badge">{verified_badge}</span>
        <span class="atlas-status-badge">{wake_badge}</span>
      </div>
      <div class="atlas-status-grid">
        <div class="atlas-metric">
          <div class="atlas-metric-label">Verified User</div>
          <div class="atlas-metric-value">{user_label}</div>
        </div>
        <div class="atlas-metric">
          <div class="atlas-metric-label">Ready Window</div>
          <div class="atlas-metric-value">{ready_left} s</div>
        </div>
        <div class="atlas-metric">
          <div class="atlas-metric-label">Intent</div>
          <div class="atlas-metric-value">{intent_label}</div>
        </div>
        <div class="atlas-metric">
          <div class="atlas-metric-label">Intent Confidence</div>
          <div class="atlas-metric-value">{intent_conf}</div>
        </div>
      </div>
    </section>
    """


def build_dorm_state_html(state: dict) -> str:
    control_state = state["control_state"]
    room = control_state.get("room", "bedroom").title()
    light_state = control_state.get("light", "off").title()
    brightness = control_state.get("brightness", 0)
    blinds = control_state.get("blinds", "open").title()
    temperature = control_state.get("temperature_c", 20)
    scene = control_state.get("scene", "relax").title()
    timers = control_state.get("timers", [])
    timer_count = len(timers)
    if timer_count:
        timer_text = ", ".join(timer["name"] for timer in timers[:3])
    else:
        timer_text = "No active timers"

    return f"""
    <section class="atlas-dorm-grid">
      <div class="atlas-dorm-card atlas-dorm-wide atlas-dorm-hero">
        <div class="atlas-dorm-label">Room</div>
        <div class="atlas-dorm-value">{room}</div>
        <div class="atlas-dorm-subvalue">Scene: {scene} • Temp: {temperature}&deg;C</div>
      </div>
      <div class="atlas-dorm-card">
        <div class="atlas-dorm-label">Light</div>
        <div class="atlas-dorm-value">{light_state}</div>
      </div>
      <div class="atlas-dorm-card">
        <div class="atlas-dorm-label">Blinds</div>
        <div class="atlas-dorm-value">{blinds}</div>
      </div>
      <div class="atlas-dorm-card atlas-dorm-wide">
        <div class="atlas-dorm-label">Brightness</div>
        <div class="atlas-progress-track"><div class="atlas-progress-bar" style="width: {brightness}%"></div></div>
        <div class="atlas-progress-label">{brightness}%</div>
      </div>
      <div class="atlas-dorm-card">
        <div class="atlas-dorm-label">Temperature</div>
        <div class="atlas-dorm-value">{temperature}&deg;C</div>
      </div>
      <div class="atlas-dorm-card">
        <div class="atlas-dorm-label">Scene</div>
        <div class="atlas-dorm-value">{scene}</div>
      </div>
      <div class="atlas-dorm-card atlas-dorm-wide">
        <div class="atlas-dorm-label">Timers</div>
        <div class="atlas-dorm-value">{timer_count}</div>
        <div class="atlas-dorm-subvalue">{timer_text}</div>
      </div>
    </section>
    """


def build_stage_hint_html(state: dict) -> str:
    stage_index = get_current_stage_index(state)
    stage_label = PIPELINE_STEPS[stage_index][1]
    next_label = PIPELINE_STEPS[stage_index + 1][1] if stage_index + 1 < len(PIPELINE_STEPS) else "Demo Complete"
    hints = [
        (
            "Start with User Identification",
            "Verify Atlas with audio or use the code bypass so the wake-word stage unlocks.",
            "Use voice verification or the code path `Nair`, `Adjmal`, or `Sharma`.",
        ),
        (
            "Unlock Wake Word",
            "Atlas is verified. Trigger the wake word or use the typed bypass to move into command mode.",
            "Say the wake word or use the code `Hey Atlas`.",
        ),
        (
            "Capture the Command",
            "Record a spoken command or type one of the demo examples to drive the rest of the pipeline.",
            "Try a movie, weather, dorm, or out-of-scope example.",
        ),
        (
            "Interpret the Request",
            "Run intent detection to classify the transcript and extract structured slot values.",
            "Confirm the predicted intent and extracted slot values before continuing.",
        ),
        (
            "Execute the Action",
            "Fulfillment calls the weather API, uses movie knowledge, or updates the dorm state.",
            "Check the action result and dorm state before generating the response.",
        ),
        (
            "Generate the Final Response",
            "The pipeline is ready to produce the final answer and synthesize audio playback.",
            "Generate the answer, then play the TTS output to complete the flow.",
        ),
    ]
    title, body, prompt = hints[stage_index]
    return f"""
    <section class="atlas-hint-card">
      <div class="atlas-hint-kicker">Current Step</div>
      <div class="atlas-hint-stepno">Step {stage_index + 1} of {len(PIPELINE_STEPS)} • {stage_label}</div>
      <h3>{title}</h3>
      <p>{body}</p>
      <div class="atlas-hint-prompt">{prompt}</div>
      <div class="atlas-hint-next">Unlocks next: {next_label}</div>
    </section>
    """


def get_pipeline_ui_updates(state: dict, raw_status_text: str, ready_left: int):
    current_stage_index = get_current_stage_index(state)
    return (
        build_status_overview_html(state, ready_left),
        raw_status_text,
        build_pipeline_timeline_html(state),
        build_dorm_state_html(state),
        build_stage_hint_html(state),
        gr.update(visible=current_stage_index == 0),
        gr.update(visible=current_stage_index == 1),
        gr.update(visible=current_stage_index == 2),
        gr.update(visible=current_stage_index == 3),
        gr.update(visible=current_stage_index == 4),
        gr.update(visible=current_stage_index >= 5),
        gr.update(visible=current_stage_index >= 4),
    )
