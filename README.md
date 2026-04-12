---
title: Atlas Virtual Assistant
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---

# Atlas Virtual Assistant

Atlas is a course project voice assistant for movie information and smart dorm control. The live implementation follows the required project pipeline:

1. User verification / identification
2. Wake word detection
3. Automatic speech recognition
4. Intent detection + slot filling
5. Fulfillment
6. Answer generation
7. Text-to-speech

The app includes bypasses for every stage where the project required one:

- verification code bypass
- typed wake-word bypass
- typed transcript bypass
- manual intent / slots bypass
- manual fulfillment JSON bypass
- manual answer bypass

## Live entry point

The supported application path is the Gradio app in [app.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/app.py:1).

Core live modules:

- [app.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/app.py:1): configuration and runtime wiring
- [atlas_voice.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/atlas_voice.py:1): dataset loading, user verification, wake-word, and Whisper ASR helpers
- [atlas_actions.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/atlas_actions.py:1): pipeline logic and UI callbacks
- [atlas_state.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/atlas_state.py:1): state and stage helpers
- [atlas_fulfillment.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/atlas_fulfillment.py:1): weather, movie, timer, and smart dorm fulfillment
- [atlas_answer_generation.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/atlas_answer_generation.py:1): template-based answer generation
- [atlas_tts.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/atlas_tts.py:1): speech synthesis
- [atlas_ui.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/atlas_ui.py:1): Gradio UI

Intent training and inference:

- [intent_data/atlas_demo_intents.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/intent_data/atlas_demo_intents.py:1): intent schema and seed examples
- [intent_data/atlas_dataset_builder.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/intent_data/atlas_dataset_builder.py:1): dataset generation
- [intent_data/train_joint_intent_slot.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/intent_data/train_joint_intent_slot.py:1): DistilBERT joint model training
- [intent_data/intent_inference.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/intent_data/intent_inference.py:1): runtime inference

## Legacy files

The following files are prototype exports from earlier Colab work. They are not part of the live app path and should not be treated as the current implementation:

- [virtual_assistant_project_.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/virtual_assistant_project_.py:1)
- [virtual_assistant_project_voice_verification_draft.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/virtual_assistant_project_voice_verification_draft.py:1)

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Optional movie API credentials:

- `TMDB_API_KEY`
- `TMDB_BEARER_TOKEN`

If TMDB credentials are missing, Atlas still works using the built-in fallback movie data for the demo examples.

## Run the app

```bash
python app.py
```

The app will:

- download the Hugging Face dataset `yadjm084/atlas-voice-data`
- load enrollment audio for user verification
- load wake-word weights
- load the Whisper `tiny` ASR model
- load the trained joint intent-slot model from `intent_data/model_artifacts/atlas_joint_intent_slot`

## Smoke tests

Typed demo path:

```bash
python demo_smoke_test.py --skip-tts
```

Typed path plus real-audio verification and wake-word checks:

```bash
python demo_smoke_test.py --skip-tts --audio-gates
```

## Demo and UI guide

Use the step-by-step operator guide here:

- [docs/ATLAS_UI_DEMO_GUIDE.md](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/docs/ATLAS_UI_DEMO_GUIDE.md:1)

That guide explains what becomes visible at each step, what the user clicks, what Atlas should do, and how to use the bypass paths during a live demo.
