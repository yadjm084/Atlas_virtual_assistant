"""Train the Atlas joint intent classification and slot filling model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from intent_data.joint_intent_slot_model import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    JointIntentSlotModel,
    save_metadata,
)

try:
    from seqeval.metrics import f1_score as seq_f1_score
except Exception:
    seq_f1_score = None


class JointDataset(Dataset):
    def __init__(self, encodings, slot_labels, intent_labels):
        self.encodings = encodings
        self.slot_labels = slot_labels
        self.intent_labels = intent_labels

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx].clone().detach(),
            "attention_mask": self.encodings["attention_mask"][idx].clone().detach(),
            "slot_labels": torch.tensor(self.slot_labels[idx], dtype=torch.long),
            "intent_label": torch.tensor(self.intent_labels[idx], dtype=torch.long),
        }

    def __len__(self):
        return len(self.intent_labels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="intent_data/atlas_training_dataset.jsonl",
        help="JSONL dataset built from atlas_dataset_builder.py",
    )
    parser.add_argument(
        "--output-dir",
        default="intent_data/model_artifacts/atlas_joint_intent_slot",
        help="Directory where model artifacts will be saved.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_annotated_sentence(sentence: str):
    tokens = []
    slots = []

    for word in sentence.split():
        if "/" in word:
            token, slot = word.split("/", 1)
        else:
            token, slot = word, "O"
        tokens.append(token)
        slots.append(slot)

    return tokens, slots


def build_examples(records):
    all_tokens = []
    all_slots = []
    all_intents = []

    for record in records:
        tokens, slots = parse_annotated_sentence(record["annotated"])
        all_tokens.append(tokens)
        all_slots.append(slots)
        all_intents.append(record["intent"])

    return all_tokens, all_slots, all_intents


def build_label_maps(all_slots, all_intents):
    unique_slots = sorted({slot for seq in all_slots for slot in seq})
    unique_intents = sorted(set(all_intents))

    slot2id = {slot: idx for idx, slot in enumerate(unique_slots)}
    id2slot = {idx: slot for slot, idx in slot2id.items()}
    intent2id = {intent: idx for idx, intent in enumerate(unique_intents)}
    id2intent = {idx: intent for intent, idx in intent2id.items()}

    return slot2id, id2slot, intent2id, id2intent


def align_slot_labels(all_tokens, all_slots, encodings, slot2id):
    aligned_labels = []

    for i in range(len(all_tokens)):
        word_ids = encodings.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(slot2id[all_slots[i][word_id]])
            else:
                label_ids.append(-100)
            previous_word_id = word_id

        aligned_labels.append(label_ids)

    return aligned_labels


def subset(items, indices):
    return [items[i] for i in indices]


def evaluate_intent(model, data_loader, id2intent, device):
    model.eval()
    intent_true = []
    intent_pred = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent_label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs["intent_logits"], dim=1)

            intent_true.extend(intent_labels.cpu().tolist())
            intent_pred.extend(preds.cpu().tolist())

    true_names = [id2intent[i] for i in intent_true]
    pred_names = [id2intent[i] for i in intent_pred]

    accuracy = sum(t == p for t, p in zip(true_names, pred_names)) / len(true_names)
    report = classification_report(true_names, pred_names, output_dict=True, zero_division=0)

    return {
        "accuracy": accuracy,
        "report": report,
        "true_names": true_names,
        "pred_names": pred_names,
    }


def evaluate_slots(model, data_loader, id2slot, device):
    model.eval()
    all_true = []
    all_pred = []
    sequence_true = []
    sequence_pred = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            slot_labels = batch["slot_labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs["slot_logits"], dim=-1)

            for pred_seq, true_seq in zip(preds.cpu().tolist(), slot_labels.cpu().tolist()):
                seq_true = []
                seq_pred = []

                for pred_id, true_id in zip(pred_seq, true_seq):
                    if true_id == -100:
                        continue
                    all_true.append(true_id)
                    all_pred.append(pred_id)
                    seq_true.append(id2slot[true_id])
                    seq_pred.append(id2slot[pred_id])

                if seq_true:
                    sequence_true.append(seq_true)
                    sequence_pred.append(seq_pred)

    token_accuracy = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
    result = {
        "token_accuracy": token_accuracy,
    }

    if seq_f1_score is not None:
        result["seqeval_f1"] = seq_f1_score(sequence_true, sequence_pred)

    return result


def save_metrics(output_dir: Path, metrics: dict):
    path = output_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_validation_examples(output_dir: Path, records, val_idx, true_intents, pred_intents):
    examples = []
    for local_idx, global_idx in enumerate(val_idx[:20]):
        record = records[global_idx]
        examples.append(
            {
                "plain_text": record["plain_text"],
                "annotated": record["annotated"],
                "true_intent": true_intents[local_idx],
                "predicted_intent": pred_intents[local_idx],
            }
        )

    path = output_dir / "validation_examples.json"
    path.write_text(json.dumps(examples, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(dataset_path)
    all_tokens, all_slots, all_intents = build_examples(records)
    slot2id, id2slot, intent2id, id2intent = build_label_maps(all_slots, all_intents)
    intent_label_ids = [intent2id[intent] for intent in all_intents]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encodings = tokenizer(
        all_tokens,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    aligned_slot_labels = align_slot_labels(all_tokens, all_slots, encodings, slot2id)

    train_idx, val_idx = train_test_split(
        list(range(len(intent_label_ids))),
        test_size=args.val_size,
        random_state=args.seed,
        stratify=intent_label_ids,
    )

    train_encodings = {
        "input_ids": encodings["input_ids"][train_idx],
        "attention_mask": encodings["attention_mask"][train_idx],
    }
    val_encodings = {
        "input_ids": encodings["input_ids"][val_idx],
        "attention_mask": encodings["attention_mask"][val_idx],
    }

    train_dataset = JointDataset(
        train_encodings,
        subset(aligned_slot_labels, train_idx),
        subset(intent_label_ids, train_idx),
    )
    val_dataset = JointDataset(
        val_encodings,
        subset(aligned_slot_labels, val_idx),
        subset(intent_label_ids, val_idx),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointIntentSlotModel(
        num_intents=len(intent2id),
        num_slots=len(slot2id),
        model_name=args.model_name,
    )

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent_label"].to(device)
            slot_labels = batch["slot_labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                intent_labels=intent_labels,
                slot_labels=slot_labels,
            )

            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_train_loss = total_loss / max(len(train_loader), 1)
        intent_metrics = evaluate_intent(model, val_loader, id2intent, device)
        slot_metrics = evaluate_slots(model, val_loader, id2slot, device)

        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "intent_accuracy": intent_metrics["accuracy"],
            "slot_token_accuracy": slot_metrics["token_accuracy"],
        }
        if "seqeval_f1" in slot_metrics:
            epoch_summary["slot_seqeval_f1"] = slot_metrics["seqeval_f1"]

        history.append(epoch_summary)
        print(epoch_summary)

    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir / "model_state.pt")

    metadata = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "intent2id": intent2id,
        "id2intent": {str(k): v for k, v in id2intent.items()},
        "slot2id": slot2id,
        "id2slot": {str(k): v for k, v in id2slot.items()},
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "freeze_encoder": args.freeze_encoder,
        "dataset_path": str(dataset_path),
    }
    save_metadata(output_dir, metadata)

    final_intent_metrics = evaluate_intent(model, val_loader, id2intent, device)
    final_slot_metrics = evaluate_slots(model, val_loader, id2slot, device)
    metrics = {
        "history": history,
        "final_intent_accuracy": final_intent_metrics["accuracy"],
        "final_intent_report": final_intent_metrics["report"],
        "final_slot_metrics": final_slot_metrics,
    }
    save_metrics(output_dir, metrics)
    save_validation_examples(
        output_dir,
        records,
        val_idx,
        final_intent_metrics["true_names"],
        final_intent_metrics["pred_names"],
    )

    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
