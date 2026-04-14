# Wake-Word Status

This note records the current wake-word status after the local retraining pass.

## Current runtime

The app now prefers a local classifier bundle if it exists:

- model bundle: `atlas-voice-data-wav/wake_word/wake_word_classifier.pkl`
- metadata: `atlas-voice-data-wav/wake_word/wake_word_classifier.json`
- fallback: the old TensorFlow `wake_word.weights.h5`

Relevant code:

- [atlas_voice.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/atlas_voice.py:98)
- [atlas_voice.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/atlas_voice.py:266)
- [app.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/app.py:41)

## Why this changed

The original shipped wake-word weights were effectively collapsed:

- most files, including positives, scored around `0.4961`
- the live threshold was `0.5`
- recall was extremely poor

That is why the UI was frequently showing wake-word probabilities around `0.49x`.

## New local model

The new local wake-word model was trained from the existing local wake dataset using:

- local data only
- MFCC mean/std summary features
- an `ExtraTreesClassifier`
- threshold selected from out-of-fold cross-validated probabilities

Training script:

- [train_wakeword_local.py](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/train_wakeword_local.py:1)

## Cross-validated training result

Saved during training:

- accuracy: `0.7917`
- precision: `0.7500`
- recall: `0.8750`
- AUC: `0.8012`
- chosen threshold: `0.434`

Saved reports:

- [docs/wakeword_retrain_report.json](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/docs/wakeword_retrain_report.json:1)
- [docs/wakeword_diagnostic_report.json](/home/aristotle/Desktop/VA_Final_Project/Atlas_virtual_assistant/docs/wakeword_diagnostic_report.json:1)

## Runtime verification

After installing the new local model, the app runtime now reports:

- `Wake word classifier loaded from wake_word_classifier.pkl`
- threshold `0.434`

Smoke test verification passed through the actual app path:

- positive sample detected
- negative sample rejected
- near-miss sample rejected

## Useful commands

Train or retrain the local wake-word model:

```bash
python train_wakeword_local.py
```

Inspect the current local wake-word model:

```bash
python wakeword_diagnostic.py --show-files
```
