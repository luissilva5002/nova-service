"""
classify_test.py

Quick manual sanity check for the trained classifier before wiring it
into NOVA. Loads intent_classifier.joblib and lets you type sentences
to see the predicted intent, confidence, and full score breakdown.

Usage:
    python classify_test.py
    python classify_test.py "check your memory for the pcb file"
"""
import sys
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer

MODEL_PATH = Path(__file__).parent / "intent_classifier.joblib"

# Set this to whatever threshold you picked from training_report.txt's
# threshold sweep - below this, the result is forced to "chat".
CONFIDENCE_THRESHOLD = 0.6


def load():
    if not MODEL_PATH.exists():
        print(f"No trained model found at {MODEL_PATH}. Run train_intent_classifier.py first.")
        sys.exit(1)
    bundle = joblib.load(MODEL_PATH)
    embedder = SentenceTransformer(bundle["embedding_model_name"])
    return embedder, bundle["classifier"]


def classify(embedder, clf, text: str) -> dict:
    embedding = embedder.encode([text], normalize_embeddings=True)
    probs = clf.predict_proba(embedding)[0]
    classes = clf.classes_
    best_idx = probs.argmax()
    confidence = float(probs[best_idx])
    raw_intent = classes[best_idx]
    intent = raw_intent if confidence >= CONFIDENCE_THRESHOLD else "chat"
    return {
        "intent": intent,
        "raw_intent": raw_intent,
        "confidence": confidence,
        "forced_to_chat": intent != raw_intent,
        "all_scores": {c: round(float(p), 3) for c, p in sorted(zip(classes, probs), key=lambda x: -x[1])},
    }


def main():
    embedder, clf = load()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = classify(embedder, clf, text)
        print(f"text: {text!r}")
        for k, v in result.items():
            print(f"  {k}: {v}")
        return

    print("Type a sentence to classify (empty line or Ctrl+C to quit).")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        result = classify(embedder, clf, text)
        print(f"  intent={result['intent']}  raw_intent={result['raw_intent']}  "
              f"confidence={result['confidence']:.3f}  forced_to_chat={result['forced_to_chat']}")
        print(f"  all_scores: {result['all_scores']}")


if __name__ == "__main__":
    main()
