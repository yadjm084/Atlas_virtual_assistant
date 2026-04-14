# Atlas UI Demo Guide

This document explains how to operate the Atlas UI during a demo and what the user should expect to see at each stage.

## Purpose

Atlas is a guided pipeline demo. The UI is intentionally step-based so the user can move through the project requirements in order:

1. User verification
2. Wake word detection
3. Speech recognition
4. Intent detection
5. Fulfillment
6. Answer generation
7. Text-to-speech

The sidebar shows:

- `Demo Progress`: the current stage in the pipeline
- `System State`: verified user, awake/sleep state, ready countdown, current intent
- `Home Theater State`: visible once the fulfillment stage is active

The main panel shows only the current step panel that the user should work with next.

## General user story

The user opens Atlas.

What is visible:

- the hero banner
- the pipeline status sidebar
- `Step 1. User Verification`

The user completes the current step either with real audio or with the built-in bypass for that step.

When the step succeeds:

- the current step is marked as completed in the sidebar
- the next step panel becomes visible
- the stage hint card updates to explain the next action

If Atlas goes back to sleep because the ready timer expires:

- the app returns to the wake-word stage
- later pipeline panels are hidden again
- the user must wake Atlas before continuing

## Step 1. User Verification

Visible panel:

- `Step 1. User Verification`

What the user can do:

- record or upload a verification voice sample and click `Verify Voice`
- or type one of the allowed bypass codes and click `Skip Verification with Code`

Allowed codes:

- `Adjmal`
- `Nair`
- `Sharma`

Expected behavior:

- if verification succeeds, Atlas becomes `Verified`
- the verification score details are shown
- `Step 2. Wake Word` becomes visible

If verification fails:

- Atlas remains locked
- the user stays on Step 1

Reset behavior:

- clicking `Reset Verification` clears verification and all downstream progress

## Step 2. Wake Word

Visible only after successful verification.

Visible panel:

- `Step 2. Wake Word`

What the user can do:

- record or upload a wake-word audio sample and click `Detect Wake Word`
- or type the bypass code and click `Skip Wake Word with Code`

Allowed wake bypass code:

- `Hey Atlas`

Expected behavior:

- if wake-word detection succeeds, Atlas becomes `Awake`
- a ready countdown starts
- `Step 3. Speech Recognition` becomes visible

If wake-word detection fails:

- Atlas stays asleep
- the user remains on Step 2

Reset behavior:

- clicking `Reset Wake Word` sends Atlas back to the wake-word stage and clears downstream command state

## Step 3. Speech Recognition

Visible only after wake succeeds.

Visible panel:

- `Step 3. Speech Recognition`

What the user can do:

- record or upload command audio and click `Run ASR`
- or type a command and click `Use Typed Sentence`
- or use one of the quick example buttons

Example buttons:

- `Weather Example`
- `Movie Example`
- `Home Theater Example`
- `Out-of-Scope Example`

Expected behavior:

- the transcript box is filled
- `Step 4. Intent Detection` becomes visible

Recommended demo path:

- during a live presentation, typed input is the fastest and most reliable route
- during a technical walk-through, audio ASR can be demonstrated for realism

## Step 4. Intent Detection

Visible after a transcript exists.

Visible panel:

- `Step 4. Intent Detection`

What the user can do:

- click `Detect Intent` to run the trained joint intent-slot model
- or open the manual bypass accordion and use `Use Manual Intent / Slots`

Expected behavior:

- the intent textbox is filled with the predicted intent
- the slots textbox is filled with extracted slot JSON
- `Step 5. Action / Fulfillment` becomes visible

What the user should verify:

- the predicted intent is correct
- the extracted slots are reasonable before continuing

## Step 5. Action / Fulfillment

Visible after intent detection.

Visible panel:

- `Step 5. Action / Fulfillment`

What the user can do:

- click `Run Action`
- or open the manual bypass accordion and use `Use Manual API Result`

What Atlas can fulfill:

- mandatory flows:
  - greetings
  - goodbye
  - timer
  - weather
- specialized domain:
  - movie information
- interactive control:
  - home theater simulation

Expected behavior:

- the fulfillment output JSON is shown
- if the command affects the home theater, the `Home Theater State` sidebar card updates
- `Step 6. Assistant Response` becomes visible

Examples of what should happen:

- weather: a structured weather result is produced
- movie query: a movie details or recommendation result is produced
- home theater control: the room state changes visibly

## Step 6. Assistant Response

Visible after fulfillment.

Visible panel:

- `Step 6. Assistant Response`

What the user can do:

- click `Generate Answer` to turn the fulfillment result into a user-facing response
- or open the manual bypass accordion and click `Use Manual Answer`
- click `Run TTS` to synthesize the generated answer

Expected behavior after `Generate Answer`:

- the `Atlas response` textbox is populated with natural-language output

Expected behavior after `Run TTS`:

- the TTS status box updates
- an audio player becomes available with generated speech

Reset behavior:

- `Reset All` returns the UI to the initial locked state

## Recommended demo flows

### Flow A: Weather

1. Verify with code `Nair`
2. Skip wake with code `Hey Atlas`
3. Click `Weather Example`
4. Click `Use Typed Sentence`
5. Click `Detect Intent`
6. Click `Run Action`
7. Click `Generate Answer`
8. Click `Run TTS`

Expected result:

- weather intent and slots are shown
- a weather response is generated
- a spoken answer is produced

### Flow B: Movie domain

1. Verify with code `Nair`
2. Skip wake with code `Hey Atlas`
3. Click `Movie Example`
4. Click `Use Typed Sentence`
5. Click `Detect Intent`
6. Click `Run Action`
7. Click `Generate Answer`
8. Optional: click `Run TTS`

Expected result:

- a movie-information intent is detected
- movie fulfillment returns data from TMDB or the fallback movie database
- Atlas generates a natural-language answer

### Flow C: Home theater control

1. Verify with code `Nair`
2. Skip wake with code `Hey Atlas`
3. Click `Home Theater Example`
4. Click `Use Typed Sentence`
5. Click `Detect Intent`
6. Click `Run Action`
7. Click `Generate Answer`

Expected result:

- a home-theater control intent is detected
- the home theater visual state changes in the sidebar
- the response confirms the state change

### Flow D: Out-of-scope

1. Verify with code `Nair`
2. Skip wake with code `Hey Atlas`
3. Click `Out-of-Scope Example`
4. Click `Use Typed Sentence`
5. Click `Detect Intent`
6. Click `Run Action`
7. Click `Generate Answer`

Expected result:

- the intent is `OOS`
- Atlas explains the request is outside the supported scope

## Notes for presenters

- The typed bypass path is the safest path for a live class demo.
- The real audio path is still available and can be shown for user verification, wake-word detection, and Whisper ASR.
- If an API call fails during the live demo, use the manual bypass at the relevant step instead of abandoning the run.
- The home theater control path is the most visually clear flow because it updates the sidebar state card.
