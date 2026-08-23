"""
train_intent_classifier.py

Trains NOVA's intent classifier: sentence-embeddings (MiniLM) ->
LogisticRegression over intent categories. No GPU needed - the whole
thing runs in well under a minute on CPU.

Usage:
    python train_intent_classifier.py

Outputs (in this folder):
    intent_classifier.joblib   - the trained sklearn model
    training_report.txt        - accuracy report + per-example
                                  confidence dump for wrong predictions
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).parent / "intent_dataset.jsonl"
MODEL_OUT_PATH = Path(__file__).parent / "intent_classifier.joblib"
REPORT_OUT_PATH = Path(__file__).parent / "training_report.txt"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping bad JSONL line {line_no}: {exc}", file=sys.stderr)
                continue
            if "text" not in row or "intent" not in row:
                print(f"Skipping line {line_no} - missing 'text' or 'intent' key", file=sys.stderr)
                continue
            texts.append(row["text"])
            labels.append(row["intent"])
    return texts, labels


def main() -> None:
    print(f"Loading dataset from {DATASET_PATH} ...")
    texts, labels = load_dataset(DATASET_PATH)
    print(f"Loaded {len(texts)} examples across {len(set(labels))} intents: {sorted(set(labels))}")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads ~90MB) ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Encoding examples ...")
    X = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    y = np.array(labels)

    X_train, X_val, y_train, y_val, texts_train, texts_val = train_test_split(
        X, y, texts, test_size=0.2, stratify=y, random_state=42
    )

    print("Training LogisticRegression classifier ...")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    val_probs = clf.predict_proba(X_val)
    val_confidences = val_probs.max(axis=1)

    report_lines = []
    report_lines.append("=== Classification report (held-out validation split) ===")
    report_lines.append(classification_report(y_val, val_preds))

    report_lines.append("\n=== Wrong predictions (check these first - confidence should be LOW) ===")
    n_wrong = 0
    for text, true_label, pred_label, conf in zip(texts_val, y_val, val_preds, val_confidences):
        if true_label != pred_label:
            n_wrong += 1
            report_lines.append(f"conf={conf:.2f}  true={true_label!r}  pred={pred_label!r}  text={text!r}")
    if n_wrong == 0:
        report_lines.append("(none - every validation example was classified correctly)")

    report_lines.append("\n=== Confidence threshold sweep ===")
    report_lines.append(f"{'threshold':>10} {'kept':>6} {'forced_to_chat':>15} {'acc_on_kept':>12}")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        kept_mask = val_confidences >= threshold
        forced = int((~kept_mask).sum())
        kept = int(kept_mask.sum())
        acc = float((val_preds[kept_mask] == y_val[kept_mask]).mean()) if kept > 0 else float("nan")
        report_lines.append(f"{threshold:>10.1f} {kept:>6} {forced:>15} {acc:>12.3f}")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    REPORT_OUT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nFull report written to {REPORT_OUT_PATH}")

    # Retrain on ALL data (train + val) for the final artifact, now that
    # we've measured generalization on the held-out split above.
    print("\nRetraining on full dataset for final artifact ...")
    final_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    final_clf.fit(X, y)

    joblib.dump(
        {
            "classifier": final_clf,
            "embedding_model_name": EMBEDDING_MODEL_NAME,
            "classes": list(final_clf.classes_),
        },
        MODEL_OUT_PATH,
    )
    print(f"Saved trained classifier to {MODEL_OUT_PATH}")
    print("\nDone. Review training_report.txt, especially the wrong-predictions section,")
    print("before picking a confidence threshold to use in NOVA.")


if __name__ == "__main__":
    main()
