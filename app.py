import gradio as gr
import json


# This function creates the initial state of the assistant.
# We store all important information here so it can be reused
# across the different pipeline steps.
def init_state():
    return {
        "verified": False,
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
# We will display it in the interface so the user can always see
# the current status of the assistant.
def get_status_text(state):
    return (
        f"Verified: {state['verified']}\n"
        f"Awake: {state['awake']}\n"
        f"Intent: {state['intent'] if state['intent'] else 'None'}"
    )


# -------------------------
# Placeholder backend functions
# -------------------------

def do_verify(audio, state):
    # Simulate user verification success
    state["verified"] = True
    return "User verified successfully (placeholder).", state, get_status_text(state)


def skip_verify(state):
    # Simulate skipping verification
    state["verified"] = True
    return "Verification skipped.", state, get_status_text(state)


def do_wake(audio, state):
    # Simulate wake word detection success
    state["awake"] = True
    return "Wake word detected (placeholder).", state, get_status_text(state)


def skip_wake(state):
    # Simulate skipping wake word detection
    state["awake"] = True
    return "Wake word detection skipped.", state, get_status_text(state)


def do_asr(audio, state):
    # Simulate speech-to-text output
    fake_text = "turn on the lamp"
    state["transcript"] = fake_text
    return fake_text, state, get_status_text(state)


def do_intent(transcript, state):
    # Simulate intent detection and slot extraction
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
    # This function lets the user bypass the intent module
    # by manually entering the intent and the slots.
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


def do_fulfillment(state):
    # Simulate fulfillment / API call / device control

    if state["intent"] == "control_device":
        device = state["slots"].get("device", "lamp")
        action = state["slots"].get("action", "on")

        # Simulate changing the control system state
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
    # This function lets the user bypass the fulfillment step
    # and manually enter a fake API result.
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


def do_answer(state):
    # Simulate answer generation based on the result of fulfillment
    if state["api_result"]:
        answer = f"Assistant response: {state['api_result'].get('message', 'Done.')}"
    else:
        answer = "Assistant response: Placeholder answer generated."

    state["answer_text"] = answer
    return answer, state, get_status_text(state)


def do_tts(state):
    # Simulate text-to-speech output
    return f"TTS would say: {state['answer_text']}"


def reset_all():
    # Reset everything to the initial state
    state = init_state()

    return (
        state,
        get_status_text(state),  # assistant status
        "",   # verification output
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


# -------------------------
# Gradio interface
# -------------------------

with gr.Blocks(title="Virtual Assistant") as demo:
    gr.Markdown("# Virtual Assistant")
    gr.Markdown("Starter version with placeholder functions only.")

    # Shared assistant state
    state = gr.State(init_state())

    # Top row: audio input + assistant status + control system state
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Voice Input")

        assistant_status = gr.Textbox(
            label="Assistant Status",
            value=get_status_text(init_state()),
            lines=4
        )

        control_box = gr.Textbox(
            label="Control System State",
            value=json.dumps(init_state()["control_state"], indent=2),
            lines=8
        )

    # Verification and wake word outputs
    with gr.Row():
        verify_output = gr.Textbox(label="Verification Output")
        wake_output = gr.Textbox(label="Wake Word Output")

    # Transcript output
    transcript_box = gr.Textbox(label="Transcript", lines=3)

    # Intent and slots outputs
    with gr.Row():
        intent_box = gr.Textbox(label="Detected Intent")
        slots_box = gr.Textbox(label="Detected Slots", lines=6)

    # Manual bypass section
    with gr.Accordion("Manual / Bypass Options", open=True):
        manual_intent = gr.Textbox(label="Manual Intent", value="control_device")

        manual_slots = gr.Textbox(
            label="Manual Slots (JSON)",
            lines=6,
            value='{\n  "device": "lamp",\n  "action": "on"\n}'
        )

        manual_api_result = gr.Textbox(
            label="Manual API Result (JSON)",
            lines=6,
            value='{\n  "status": "success",\n  "message": "lamp turned on"\n}'
        )

    # Remaining outputs
    api_box = gr.Textbox(label="Fulfillment / API Output", lines=8)
    answer_box = gr.Textbox(label="Final Answer", lines=3)
    tts_output = gr.Textbox(label="TTS Output", lines=2)

    gr.Markdown("## Pipeline Controls")

    with gr.Row():
        btn_verify = gr.Button("Run Verification")
        btn_skip_verify = gr.Button("Skip Verification")

    with gr.Row():
        btn_wake = gr.Button("Run Wake Word")
        btn_skip_wake = gr.Button("Skip Wake Word")

    with gr.Row():
        btn_asr = gr.Button("Run ASR")
        btn_intent = gr.Button("Run Intent Detection")
        btn_manual_intent = gr.Button("Use Manual Intent / Slots")

    with gr.Row():
        btn_fulfill = gr.Button("Run Fulfillment")
        btn_manual_api = gr.Button("Use Manual API Result")

    with gr.Row():
        btn_answer = gr.Button("Generate Answer")
        btn_tts = gr.Button("Run TTS")
        btn_reset = gr.Button("Reset")

    # -------------------------
    # Button connections
    # -------------------------

    btn_verify.click(
        fn=do_verify,
        inputs=[audio_input, state],
        outputs=[verify_output, state, assistant_status]
    )

    btn_skip_verify.click(
        fn=skip_verify,
        inputs=[state],
        outputs=[verify_output, state, assistant_status]
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