"""Evaluate the local wake-word model without changing app behavior.

This script stays local-first:
- it does not import app.py
- it does not download datasets from Hugging Face
- it evaluates the current wake-word weights on the local wake-word dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from atlas_voice import (
    extract_mfcc,
    get_wake_threshold,
    get_wake_weights_path,
    load_and_preprocess_audio,
    load_wake_model,
    predict_wake_word,
)


LABEL_MAP = {
    "positive": 1,
    "near": 0,
    "other": 0,
}


def find_local_dataset_root(base_dir: Path) -> Path:
    candidates = [
        base_dir / "atlas-voice-data-wav",
        base_dir / "atlas-voice-data",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No local wake-word dataset root was found.")


def collect_rows(dataset_root: Path, model, threshold: float) -> list[dict]:
    rows: list[dict] = []
    wake_root = dataset_root / "wake_word"
    for label in ("positive", "near", "other"):
        folder = wake_root / label
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".wav", ".m4a"}:
                continue
            result = predict_wake_word(
                audio_path=str(path),
                model=model,
                threshold=threshold,
            )
            rows.append(
                {
                    "label": label,
                    "target": LABEL_MAP[label],
                    "file": path.name,
                    "probability": result["probability_positive"],
                    "detected": result["wake_detected"],
                }
            )
    return rows


def summarize_by_label(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for label in ("positive", "near", "other"):
        subset = [row["probability"] for row in rows if row["label"] == label]
        summary[label] = {
            "count": len(subset),
            "min_probability": min(subset),
            "max_probability": max(subset),
            "avg_probability": mean(subset),
        }
    return summary


def metrics_at_threshold(rows: list[dict], threshold: float) -> dict:
    tp = tn = fp = fn = 0
    for row in rows:
        pred = 1 if row["probability"] >= threshold else 0
        target = row["target"]
        if target == 1 and pred == 1:
            tp += 1
        elif target == 1 and pred == 0:
            fn += 1
        elif target == 0 and pred == 1:
            fp += 1
        else:
            tn += 1

    total = len(rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def find_best_accuracy_threshold(rows: list[dict]) -> dict:
    thresholds = sorted({row["probability"] for row in rows})
    best_metrics = None
    for threshold in thresholds:
        current = metrics_at_threshold(rows, threshold)
        if best_metrics is None:
            best_metrics = current
            continue
        if current["accuracy"] > best_metrics["accuracy"]:
            best_metrics = current
            continue
        if current["accuracy"] == best_metrics["accuracy"] and current["recall"] > best_metrics["recall"]:
            best_metrics = current
    return best_metrics


def get_sample_feature_shape(dataset_root: Path) -> tuple[str, list[int]]:
    sample = sorted((dataset_root / "wake_word" / "positive").glob("*.wav"))[0]
    signal, sr = load_and_preprocess_audio(
        file_path=str(sample),
        target_sr=16000,
        target_length=32000,
    )
    mfcc = extract_mfcc(signal, sr, n_mfcc=13)
    return sample.name, list(mfcc.shape)


def build_report(base_dir: Path, threshold: float | None) -> dict:
    dataset_root = find_local_dataset_root(base_dir)
    weights_path = get_wake_weights_path(dataset_root)
    if threshold is None:
        threshold = get_wake_threshold(weights_path, default=0.5)
    model, status = load_wake_model(weights_path)
    if model is None:
        raise RuntimeError(status)

    rows = collect_rows(dataset_root, model, threshold)
    sample_name, sample_shape = get_sample_feature_shape(dataset_root)

    return {
        "dataset_root": str(dataset_root),
        "weights_path": str(weights_path),
        "model_status": status,
        "threshold": threshold,
        "sample_feature_shape": {
            "file": sample_name,
            "mfcc_shape": sample_shape,
        },
        "summary_by_label": summarize_by_label(rows),
        "metrics_at_current_threshold": metrics_at_threshold(rows, threshold),
        "best_accuracy_threshold": find_best_accuracy_threshold(rows),
        "rows": rows,
    }


def print_report(report: dict, show_files: bool) -> None:
    current = report["metrics_at_current_threshold"]
    best = report["best_accuracy_threshold"]

    print("Wake-word diagnostic")
    print(f"dataset_root: {report['dataset_root']}")
    print(f"weights_path: {report['weights_path']}")
    print(f"model_status: {report['model_status']}")
    print(
        "sample_feature_shape: "
        f"{report['sample_feature_shape']['file']} -> {tuple(report['sample_feature_shape']['mfcc_shape'])}"
    )
    print("")

    for label, stats in report["summary_by_label"].items():
        print(
            f"{label}: count={stats['count']} "
            f"min={stats['min_probability']:.4f} "
            f"max={stats['max_probability']:.4f} "
            f"avg={stats['avg_probability']:.4f}"
        )

    print("")
    print(
        "current_threshold: "
        f"{current['threshold']:.4f} "
        f"accuracy={current['accuracy']:.4f} "
        f"precision={current['precision']:.4f} "
        f"recall={current['recall']:.4f} "
        f"specificity={current['specificity']:.4f} "
        f"tp={current['tp']} tn={current['tn']} fp={current['fp']} fn={current['fn']}"
    )
    print(
        "best_accuracy_threshold: "
        f"{best['threshold']:.6f} "
        f"accuracy={best['accuracy']:.4f} "
        f"precision={best['precision']:.4f} "
        f"recall={best['recall']:.4f} "
        f"specificity={best['specificity']:.4f} "
        f"tp={best['tp']} tn={best['tn']} fp={best['fp']} fn={best['fn']}"
    )

    if show_files:
        print("")
        print("per_file_scores:")
        for row in report["rows"]:
            print(
                f"{row['label']:>8}  {row['probability']:.4f}  "
                f"detected={str(row['detected']):<5}  {row['file']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--show-files", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    report = build_report(base_dir, args.threshold)
    print_report(report, show_files=args.show_files)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
