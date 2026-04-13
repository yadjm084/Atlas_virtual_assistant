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
    candidates = [
        base_dir / "intent_data" / "model_artifacts" / "atlas_joint_intent_slot",
        base_dir / "intent_data" / "model_artifacts" / "smoke_test",
    ]

    for path in candidates:
        if (path / "model_state.pt").exists() and (path / "metadata.json").exists():
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


class IntentPredictor:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, metadata = load_model_from_artifacts(self.artifacts_dir, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.artifacts_dir)
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
        intent = self.intent_id2label[intent_id]
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

        # Pad defensively if tokenization edge cases produce fewer labels than words.
        while len(word_slot_labels) < len(words):
            word_slot_labels.append("O")

        slots = _bio_to_slots(words, word_slot_labels[: len(words)])

        return IntentPrediction(
            intent=intent,
            intent_confidence=intent_confidence,
            slots=slots,
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
