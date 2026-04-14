"""Train a local wake-word classifier from the local wake dataset.

This script:
- uses only local dataset files
- trains a small tree-based classifier on MFCC summary features
- selects a threshold from out-of-fold probabilities
- saves a local model bundle that the app can use directly
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from atlas_voice import (
    extract_mfcc,
    get_wake_classifier_metadata_path,
    get_wake_classifier_path,
    get_wake_weights_path,
    load_and_preprocess_audio,
    mfcc_to_stats_vector,
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
    raise FileNotFoundError("No local dataset root found.")


def load_dataset(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows: list[dict] = []
    feature_rows: list[np.ndarray] = []
    target_rows: list[int] = []
    wake_root = dataset_root / "wake_word"

    for label, target in LABEL_MAP.items():
        folder = wake_root / label
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".wav", ".m4a"}:
                continue
            signal, sr = load_and_preprocess_audio(
                file_path=str(path),
                target_sr=16000,
                target_length=32000,
            )
            mfcc = extract_mfcc(signal, sr, n_mfcc=13)
            feature_rows.append(mfcc_to_stats_vector(mfcc))
            target_rows.append(target)
            rows.append(
                {
                    "label": label,
                    "target": target,
                    "file": path.name,
                }
            )

    return np.vstack(feature_rows), np.array(target_rows, dtype=int), rows


def choose_threshold(targets: np.ndarray, probabilities: np.ndarray) -> dict:
    thresholds = sorted({round(float(prob), 6) for prob in probabilities})
    best = None

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(targets, predictions)
        recall = recall_score(targets, predictions, zero_division=0)
        precision = precision_score(targets, predictions, zero_division=0)

        current = {
            "threshold": threshold,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
        }

        if best is None:
            best = current
            continue
        if current["accuracy"] > best["accuracy"]:
            best = current
            continue
        if current["accuracy"] == best["accuracy"] and current["recall"] > best["recall"]:
            best = current

    return best


def build_report(rows: list[dict], probabilities: np.ndarray, targets: np.ndarray, threshold_stats: dict) -> dict:
    per_label = {}
    for label in LABEL_MAP:
        subset = [float(probabilities[idx]) for idx, row in enumerate(rows) if row["label"] == label]
        per_label[label] = {
            "count": len(subset),
            "min_probability": min(subset),
            "max_probability": max(subset),
            "avg_probability": mean(subset),
        }

    predictions = (probabilities >= threshold_stats["threshold"]).astype(int)
    return {
        "cross_validated_auc": roc_auc_score(targets, probabilities),
        "threshold": threshold_stats["threshold"],
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "summary_by_label": per_label,
        "rows": [
            {
                **row,
                "probability": float(probabilities[idx]),
                "detected": bool(predictions[idx]),
            }
            for idx, row in enumerate(rows)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    dataset_root = find_local_dataset_root(base_dir)
    weights_path = get_wake_weights_path(dataset_root)
    classifier_path = get_wake_classifier_path(weights_path)
    metadata_path = get_wake_classifier_metadata_path(weights_path)

    X, y, rows = load_dataset(dataset_root)

    estimator = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    oof_probabilities = cross_val_predict(
        estimator,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    threshold_stats = choose_threshold(y, oof_probabilities)
    report = build_report(rows, oof_probabilities, y, threshold_stats)

    estimator.fit(X, y)

    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    with classifier_path.open("wb") as f:
        pickle.dump(estimator, f)

    metadata = {
        "model_kind": "sklearn_mfcc_stats_extra_trees",
        "feature_type": "mfcc_mean_std",
        "target_sr": 16000,
        "target_length": 32000,
        "n_mfcc": 13,
        "threshold": threshold_stats["threshold"],
        "cv_folds": args.cv_folds,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "metrics": {
            "accuracy": report["accuracy"],
            "precision": report["precision"],
            "recall": report["recall"],
            "cross_validated_auc": report["cross_validated_auc"],
        },
        "summary_by_label": report["summary_by_label"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_path = base_dir / "docs" / "wakeword_retrain_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Local wake-word training complete.")
    print(f"dataset_root: {dataset_root}")
    print(f"classifier_path: {classifier_path}")
    print(f"metadata_path: {metadata_path}")
    print(f"report_path: {report_path}")
    print(
        "cross_validated_metrics: "
        f"accuracy={report['accuracy']:.4f} "
        f"precision={report['precision']:.4f} "
        f"recall={report['recall']:.4f} "
        f"auc={report['cross_validated_auc']:.4f} "
        f"threshold={threshold_stats['threshold']:.6f}"
    )


if __name__ == "__main__":
    main()
