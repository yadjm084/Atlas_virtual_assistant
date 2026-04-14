"""Reusable joint intent classification and slot filling model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel


DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 32


class JointIntentSlotModel(nn.Module):
    """A DistilBERT encoder with separate intent and slot heads."""

    def __init__(self, num_intents: int, num_slots: int, model_name: str = DEFAULT_MODEL_NAME):
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        self.slot_classifier = nn.Linear(hidden_size, num_slots)

    def forward(
        self,
        input_ids,
        attention_mask,
        intent_labels=None,
        slot_labels=None,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]

        intent_logits = self.intent_classifier(cls_output)
        slot_logits = self.slot_classifier(sequence_output)

        loss = None
        if intent_labels is not None and slot_labels is not None:
            intent_loss_fn = nn.CrossEntropyLoss()
            slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

            intent_loss = intent_loss_fn(intent_logits, intent_labels)
            slot_loss = slot_loss_fn(
                slot_logits.view(-1, slot_logits.shape[-1]),
                slot_labels.view(-1),
            )
            loss = intent_loss + slot_loss

        return {
            "loss": loss,
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
        }


def save_metadata(output_dir: Path, metadata: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def load_metadata(output_dir: Path) -> dict:
    metadata_path = output_dir / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_model_from_artifacts(output_dir: Path, device: str | torch.device = "cpu"):
    output_dir = Path(output_dir)

    # -------- NEW: support HF dataset format --------
    hf_model_path = output_dir / "intent_model.pt"
    hf_labels_path = output_dir / "intent_labels.json"

    if hf_model_path.exists() and hf_labels_path.exists():
        print("Loading model using HF dataset format...")

        labels_data = json.loads(hf_labels_path.read_text(encoding="utf-8"))

        metadata = {
            "intent2id": labels_data["intent2id"],
            "slot2id": labels_data["slot2id"],
            "id2intent": labels_data["id2intent"],
            "id2slot": labels_data["id2slot"],
            "model_name": DEFAULT_MODEL_NAME,
            "max_length": DEFAULT_MAX_LENGTH,
        }

        model = JointIntentSlotModel(
            num_intents=len(metadata["intent2id"]),
            num_slots=len(metadata["slot2id"]),
            model_name=metadata["model_name"],
        )

        state_dict = torch.load(hf_model_path, map_location=device)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        model.to(device)
        model.eval()

        return model, metadata

    # -------- OLD FORMAT --------
    print("Loading model using legacy artifacts format...")

    metadata = load_metadata(output_dir)

    model = JointIntentSlotModel(
        num_intents=len(metadata["intent2id"]),
        num_slots=len(metadata["slot2id"]),
        model_name=metadata["model_name"],
    )

    state_dict = torch.load(output_dir / "model_state.pt", map_location=device)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    model.to(device)
    model.eval()

    return model, metadata