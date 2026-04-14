"""Intent-model loading and inference helpers for Atlas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from intent_data.joint_intent_slot_model import load_model_from_artifacts


@dataclass
class IntentPrediction:
    intent: str
    intent_confidence: float
    slots: dict
    words: list[str]
    word_slot_labels: list[str]


def find_intent_artifacts_dir(base_dir: Path) -> Path | None:
    """
    Backward-compatible helper.

    Supports:
    1. Old local structure:
       intent_data/model_artifacts/atlas_joint_intent_slot/
         - model_state.pt
         - metadata.json

    2. HF dataset root structure:
         - intent_model.pt
         - intent_labels.json
         - intent_tokenizer/

    Returns the directory that should be passed to load_intent_predictor().
    """
    candidates = [
        base_dir / "intent_data" / "model_artifacts" / "atlas_joint_intent_slot",
        base_dir / "intent_data" / "model_artifacts" / "smoke_test",
        base_dir,
    ]

    for path in candidates:
        # Old structure
        if (path / "model_state.pt").exists() and (path / "metadata.json").exists():
            return path

        # New HF dataset structure
        if (
            (path / "intent_model.pt").exists()
            and (path / "intent_labels.json").exists()
            and (path / "intent_tokenizer").exists()
        ):
            return path

    return None


def _merge_slot_value(slots: dict, slot_name: str, value: str):
    if slot_name not in slots:
        slots[slot_name] = value
        return

    if isinstance(slots[slot_name], list):
        slots[slot_name].append(value)
    else:
        slots[slot_name] = [slots[slot_name], value]


def _bio_to_slots(words: list[str], labels: list[str]) -> dict:
    slots = {}
    current_slot_name = None
    current_tokens = []

    def flush_current():
        nonlocal current_slot_name, current_tokens
        if current_slot_name and current_tokens:
            _merge_slot_value(slots, current_slot_name, " ".join(current_tokens))
        current_slot_name = None
        current_tokens = []

    for word, label in zip(words, labels):
        if label == "O":
            flush_current()
            continue

        if "-" not in label:
            flush_current()
            continue

        prefix, slot_name = label.split("-", 1)

        if prefix == "B":
            flush_current()
            current_slot_name = slot_name
            current_tokens = [word]
        elif prefix == "I" and current_slot_name == slot_name:
            current_tokens.append(word)
        else:
            flush_current()
            current_slot_name = slot_name
            current_tokens = [word]

    flush_current()
    return slots


def map_intent(intent: str) -> str:
    mapping = {
        "Weather": "GetWeather",
        "Timer": "SetTimer",
        "SearchMovie": "MovieOverview",
    }
    return mapping.get(intent, intent)


def map_slots(slots: dict) -> dict:
    slot_mapping = {
        "location": "CITY",
        "duration": "DURATION",
        "date": "DATE",
        "title": "TITLE",
        "genre": "GENRE",
        "year": "YEAR",
        "room": "ROOM",
        "mode": "SCENE",
    }

    mapped = {}

    for key, value in slots.items():
        new_key = slot_mapping.get(key, key)

        # Context-dependent mapping for ambiguous value fields
        if key == "level":
            new_key = "BRIGHTNESS"
        elif key == "value":
            new_key = "VALUE"

        mapped[new_key] = value

    return mapped


class IntentPredictor:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Legacy structure
        legacy_model = self.artifacts_dir / "model_state.pt"
        legacy_metadata = self.artifacts_dir / "metadata.json"

        # New HF dataset structure
        hf_model = self.artifacts_dir / "intent_model.pt"
        hf_labels = self.artifacts_dir / "intent_labels.json"
        hf_tokenizer_dir = self.artifacts_dir / "intent_tokenizer"

        if legacy_model.exists() and legacy_metadata.exists():
            model_dir_for_loader = self.artifacts_dir
            tokenizer_dir = self.artifacts_dir

        elif hf_model.exists() and hf_labels.exists() and hf_tokenizer_dir.exists():
            model_dir_for_loader = self.artifacts_dir
            tokenizer_dir = hf_tokenizer_dir

        else:
            raise FileNotFoundError(
                f"Intent artifacts not found in expected format under: {self.artifacts_dir}"
            )

        self.model, metadata = load_model_from_artifacts(
            model_dir_for_loader,
            device=self.device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.max_length = metadata["max_length"]
        self.intent_id2label = {int(k): v for k, v in metadata["id2intent"].items()}
        self.slot_id2label = {int(k): v for k, v in metadata["id2slot"].items()}

    def predict(self, text: str) -> IntentPrediction:
        words = text.strip().split()
        if not words:
            raise ValueError("Empty transcript.")

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            intent_probs = torch.softmax(outputs["intent_logits"], dim=1)[0]
            slot_probs = torch.softmax(outputs["slot_logits"], dim=-1)[0]

        intent_id = int(torch.argmax(intent_probs).item())
        raw_intent = self.intent_id2label[intent_id]
        intent_confidence = float(intent_probs[intent_id].item())

        slot_pred_ids = torch.argmax(slot_probs, dim=-1).tolist()
        word_slot_labels = []
        previous_word_id = None

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id == previous_word_id:
                continue
            label = self.slot_id2label[int(slot_pred_ids[token_idx])]
            word_slot_labels.append(label)
            previous_word_id = word_id

        while len(word_slot_labels) < len(words):
            word_slot_labels.append("O")

        raw_slots = _bio_to_slots(words, word_slot_labels[: len(words)])

        mapped_intent = map_intent(raw_intent)
        mapped_slots = map_slots(raw_slots)

        return IntentPrediction(
            intent=mapped_intent,
            intent_confidence=intent_confidence,
            slots=mapped_slots,
            words=words,
            word_slot_labels=word_slot_labels[: len(words)],
        )


def load_intent_predictor(artifacts_dir: Path | None):
    if artifacts_dir is None:
        return None, "Intent model artifacts not found."

    try:
        predictor = IntentPredictor(artifacts_dir)
        return predictor, f"Intent model loaded from {artifacts_dir.name}"
    except Exception as e:
        return None, f"Intent model failed to load: {e}"