import gradio as gr
import json


# =========================================================
# STATE FUNCTIONS
# =========================================================

# This function creates the initial state of the assistant.
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
        "control_state": {
            "lamp": "off",
            "temperature": 20
        }
    }


# This function builds a readable summary of the assistant state.
def get_status_text(state):
    return (
        f"Verified: {state['verified']}\n"
        f"Verified User: {state['verified_user']}\n"
        f"Awake: {state['awake']}\n"
        f"Intent: {state['intent'] if state['intent'] else 'None'}"
    )


# =========================================================
# USER VERIFICATION FUNCTIONS
# =========================================================

# This is the placeholder verification function for audio.
# Later, you will replace the inside of this function with your
# actual speaker verification pipeline.
def do_verify(audio, state):
    # Safety check: if no audio is provided, return a message
    if audio is None:
        return "Please record or upload an audio file first.", state, get_status_text(state)

    # -----------------------------------------------------
    # PLACEHOLDER LOGIC
    # -----------------------------------------------------
    # For now, we simulate a successful verification.
    # Later, this should:
    # 1. load the audio file
    # 2. extract the speaker vector
    # 3. compare with enrolled profiles
    # 4. apply the chosen threshold
    # 5. update verified + verified_user accordingly
    # -----------------------------------------------------
    state["verified"] = True
    state["verified_user"] = "AudioUser"

    return "User verified successfully from audio (placeholder).", state, get_status_text(state)


# This function handles the verification bypass using a code.
# Based on your project logic, we will accept user names as codes.
def verify_with_code(code_input, state):
    # Normalize the input to avoid issues with spaces
    code = code_input.strip()

    valid_codes = ["Adjmal", "Nair", "Sharma"]

    if code in valid_codes:
        state["verified"] = True
        state["verified_user"] = code
        return f"Verification bypass successful. User set to {code}.", state, get_status_text(state)
    else:
        return "Invalid verification code. Use Adjmal, Nair, or Sharma.", state, get_status_text(state)


# This function resets only the verification information.
def reset_verification(state):
    state["verified"] = False
    state["verified_user"] = "None"
    return "", "", state, get_status_text(state)


# =========================================================
# WAKE WORD FUNCTIONS
# =========================================================

def do_wake(audio, state):
    # Wake word should only work after verification
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
    gr.Markdown("User verification now includes grouped controls and a bypass code.")

    state = gr.State(init_state())

    # Top row: audio input + assistant status + control system state
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

    # =====================================================
    # USER VERIFICATION SECTION
    # =====================================================
    with gr.Group():
        gr.Markdown("## User Verification")

        verify_output = gr.Textbox(
            label="Verification Output",
            lines=2
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

    # =====================================================
    # WAKE WORD SECTION
    # =====================================================
    with gr.Group():
        gr.Markdown("## Wake Word")

        wake_output = gr.Textbox(label="Wake Word Output", lines=2)

        with gr.Row():
            btn_wake = gr.Button("Run Wake Word")
            btn_skip_wake = gr.Button("Skip Wake Word")

    # =====================================================
    # ASR SECTION
    # =====================================================
    with gr.Group():
        gr.Markdown("## Speech Recognition")
        transcript_box = gr.Textbox(label="Transcript", lines=3)

        with gr.Row():
            btn_asr = gr.Button("Run ASR")

    # =====================================================
    # INTENT SECTION
    # =====================================================
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

    # =====================================================
    # FULFILLMENT SECTION
    # =====================================================
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

    # =====================================================
    # ANSWER + TTS SECTION
    # =====================================================
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

    # Verification using audio
    btn_verify.click(
        fn=do_verify,
        inputs=[audio_input, state],
        outputs=[verify_output, state, assistant_status]
    )

    # Verification bypass using code
    btn_skip_verify.click(
        fn=verify_with_code,
        inputs=[verification_code_input, state],
        outputs=[verify_output, state, assistant_status]
    )

    # Reset verification only
    btn_reset_verification.click(
        fn=reset_verification,
        inputs=[state],
        outputs=[verify_output, verification_code_input, state, assistant_status]
    )

    # Wake word
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

    # ASR
    btn_asr.click(
        fn=do_asr,
        inputs=[audio_input, state],
        outputs=[transcript_box, state, assistant_status]
    )

    # Intent
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

    # Fulfillment
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

    # Answer + TTS
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

    # Full reset
    btn_reset.click(
        fn=reset_all,
        inputs=[],
        outputs=[
            state,
            assistant_status,
            verify_output,
            verification_code_input,
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