# Atlas Slide Text

This file gives ready-to-paste text for slides where you already have screenshots of each step.

Recommended layout for each slide:

- left side: 3 to 5 short bullets from this file
- right side: your screenshot
- bottom or speaker notes: the short script

Keep the wording simple. The goal is to sound like a student explaining the system clearly, not like a research paper.

## Slide 1. Project Overview

### Text to put on the slide

- Atlas is a course virtual assistant for weather, movie information, timers, and home theater control.
- We built it as a full pipeline, not as one black-box model.
- The system follows these steps: user verification, wake word, speech recognition, intent and slot filling, fulfillment, answer generation, and text-to-speech.
- The interface is step-by-step so each project requirement can be shown clearly during the demo.

### What to say

Atlas is our end-to-end virtual assistant project. We wanted to show the full assistant pipeline from audio input all the way to spoken output. Instead of hiding everything behind one button, we made each stage visible so the professor can see exactly what the system is doing.

## Slide 2. Why The UI Is Step-By-Step

### Text to put on the slide

- The UI shows the current pipeline stage, system state, and home theater state.
- Each panel unlocks only after the previous stage succeeds.
- This makes the project easy to explain and easy to test.
- It also helps us show each course concept one by one.

### What to say

We made the UI step-based on purpose. This helps during grading because the professor can see where we are in the pipeline, what the current state is, and what changes after each step.

## Slide 3. Step 1: User Verification

### Text to put on the slide

- This step checks who is speaking before Atlas unlocks.
- We used enrollment recordings stored in the local dataset.
- From each recording, we extract MFCC features and build a speaker profile.
- We compare the new audio against saved profiles using cosine similarity.
- If the score passes the threshold, Atlas accepts the user.

### Technical line

- Main file: `atlas_voice.py`
- Method: MFCC mean and standard deviation + cosine similarity

### What to say

The first step is user verification. Before Atlas listens to commands, it checks whether the speaker matches one of the enrolled users. We used MFCC audio features, turned each speaker into a profile vector, and then compared the incoming voice sample with cosine similarity.

## Slide 4. Step 2: Wake Word Detection

### Text to put on the slide

- After verification, Atlas waits for the wake phrase `Hey Atlas`.
- The wake-word dataset has three groups: positive, near, and other.
- The live system uses a local wake-word classifier trained on MFCC summary features.
- If the wake word is detected, Atlas becomes awake and starts the ready timer.

### Technical line

- Main files: `atlas_voice.py`, `train_wakeword_local.py`
- Runtime model: local `ExtraTreesClassifier`
- Saved threshold: `0.434`

### What to say

This step is the wake-word stage. We trained a local classifier so the app can decide whether the user really said the wake phrase. We also kept near examples and unrelated examples so the model learns the difference between the true wake word and confusing audio.

## Slide 5. Step 3: Speech Recognition

### Text to put on the slide

- Once Atlas is awake, it converts the spoken command into text.
- We used Whisper for automatic speech recognition.
- The current model loaded in the app is `Whisper tiny`.
- The transcript is shown on the screen before we move to intent detection.

### Technical line

- Main files: `atlas_voice.py`, `atlas_actions.py`
- Model: `Whisper tiny`

### What to say

Here the system changes the user’s speech into text. We used Whisper tiny because it is small enough for our demo and still gives useful results. Showing the transcript on screen is important because it lets us verify what the ASR stage produced before moving forward.

## Slide 6. Step 4: Intent Detection And Slot Filling

### Text to put on the slide

- This step answers two questions: what does the user want, and which details did they mention.
- Example: `does it rain in Ottawa today`
- Intent: `GetWeather`
- Slots: `CITY = Ottawa`, `DATE = today`
- We trained one model to predict both the intent and the slot labels.

### Technical line

- Main files: `intent_data/train_joint_intent_slot.py`, `intent_data/intent_inference.py`
- Base model: `distilbert-base-uncased`
- Method: joint intent classification + BIO slot tagging

### What to say

This is the language-understanding step. The model has to identify the user’s goal and also pull out the important details. We used a joint DistilBERT model, so one model predicts both the intent and the BIO slot labels at the same time.

## Slide 7. Step 4 Extra: Supported Intents

### Text to put on the slide

- Mandatory intents: greetings, goodbye, out-of-scope, timer, weather
- Movie intents: overview, rating, director, cast, similar movies, discover by genre
- Home theater intents: light on or off, brightness, blinds, temperature, scene
- This gave us both required functions and a more interesting demo domain

### Technical line

- Total runtime intents: `18`

### What to say

We did not keep the assistant too small. Along with the required basic intents, we added a movie domain and a home theater control domain. This made the project more realistic and also gave us more cases to test.

## Slide 8. Step 5: Fulfillment

### Text to put on the slide

- Fulfillment is where Atlas actually does the requested task.
- Weather requests use Open-Meteo.
- Movie requests use TMDB if credentials are available.
- If TMDB is missing, Atlas falls back to built-in movie data.
- Home theater and timer requests update local app state.

### Technical line

- Main file: `atlas_fulfillment.py`
- APIs: Open-Meteo and TMDB

### What to say

After we know the intent and slots, the system has to do something useful. For weather, it calls Open-Meteo. For movies, it uses TMDB, or a local fallback database if the API key is missing. For timers and home theater commands, it updates the internal state of the application.

## Slide 9. Step 5 Extra: Home Theater State

### Text to put on the slide

- We used home theater control as our interactive domain.
- Atlas can change light state, brightness, blinds, temperature, and scene mode.
- The result is shown visually in the UI, not only in JSON.
- This makes the demo more concrete and easier to understand.

### What to say

One part of fulfillment that works especially well in the demo is the home theater state. Instead of only printing text, the UI shows the updated room settings. This makes it clear that the command had an actual effect inside the system.

## Slide 10. Step 6: Answer Generation

### Text to put on the slide

- After fulfillment, Atlas turns the structured result into a natural sentence.
- We used template-based answer generation.
- This makes the output stable, clear, and easy to explain.
- We did not use a large chat model for this stage.

### Technical line

- Main file: `atlas_answer_generation.py`
- Inputs: intent, slots, fulfillment result, home theater state

### What to say

At this step, we take structured output like JSON and convert it into a human-friendly response. We chose template-based generation because it is predictable and reliable for a course demo. It also makes it easy to see how the answer was built.

## Slide 11. Step 7: Text-to-Speech

### Text to put on the slide

- The final step is speaking the answer back to the user.
- We used `edge-tts` for speech synthesis.
- The current voice is `en-CA-ClaraNeural`.
- The generated audio is saved as an MP3 and played in the UI.

### Technical line

- Main file: `atlas_tts.py`
- Backend: `edge-tts`

### What to say

The last stage is text-to-speech. Once the answer text is ready, Atlas generates audio and plays it back in the interface. This completes the full assistant cycle from voice input to voice output.

## Slide 12. Demo Reliability And Bypasses

### Text to put on the slide

- We added a bypass for every major stage.
- Verification code bypass
- Typed wake-word bypass
- Typed transcript bypass
- Manual intent and slot bypass
- Manual fulfillment JSON bypass
- Manual answer bypass

### What to say

These bypasses are important for a live presentation. The point of the project is to prove the whole pipeline, so one noisy microphone recording should not ruin the demo. With the bypasses, we can still show every stage clearly.

## Slide 13. Models, APIs, And Tools Used

### Text to put on the slide

- User verification: MFCC features + cosine similarity
- Wake word: local `ExtraTreesClassifier`
- ASR: `Whisper tiny`
- Intent and slots: `distilbert-base-uncased`
- Weather API: Open-Meteo
- Movie API: TMDB
- TTS: `edge-tts`

### What to say

This slide is useful if the professor asks for the exact technical stack. It shows that Atlas is a combination of audio processing, machine learning, API integration, local state management, and speech synthesis.

## Slide 14. Evaluation And Testing

### Text to put on the slide

- We saved local model artifacts and evaluation files in the repo.
- The intent model metrics are stored in `intent_data/model_artifacts/atlas_joint_intent_slot`.
- We also created `demo_smoke_test.py` for repeatable end-to-end checks.
- The smoke test runs weather, movie, home theater, and out-of-scope flows.

### Technical line

- Current saved validation metrics:
- Intent accuracy: `97.7%`
- Slot token accuracy: `98.33%`

### What to say

We did not only build the demo interface. We also kept model artifacts, saved metrics, and created smoke tests so the project can be checked quickly before presenting. That made the final demo more reliable.

## Slide 15. Final Summary

### Text to put on the slide

- Atlas is a complete voice assistant pipeline built for the course project.
- It starts with voice, understands the request, fulfills the task, and responds with speech.
- The project combines course concepts from audio processing, wake-word detection, ASR, intent detection, slot filling, fulfillment, and answer generation.
- The UI and bypasses make the system easy to explain and stable to demo.

### What to say

In summary, Atlas is a full end-to-end assistant, not just one model or one API call. We connected all the course concepts into one working system and designed the UI so each part can be shown clearly during grading.

## Short Version If You Want Very Minimal Slides

If you want fewer words on each slide, use this pattern:

- What this step does
- How we implemented it
- Main model or API used
- Why it matters in the pipeline

Example:

- Checks who is speaking
- Uses MFCC speaker features and cosine similarity
- Main file: `atlas_voice.py`
- Needed to unlock Atlas before commands are accepted
