"""
nova/brain/intent_classifier.py

Loads and runs the trained intent classifier (MiniLM sentence embeddings
-> LogisticRegression, see classifier/train_intent_classifier.py). This
is the PRIMARY intent detection mechanism: cheap (single-digit ms on
CPU), no GGUF/llama.cpp involvement, and its softmax output is a real,
usable confidence signal - unlike a generative LLM's self-reported
"confidence" field, which small models are poorly calibrated to report
honestly.

Falls back gracefully (returns None from classify()) if the artifact
isn't present or the dependencies aren't installed, so nova/brain/
intent_router.py can defer to the Qwen3 LLM classification pass in
that case.
"""
import logging
from typing import Optional

from nova.config import INTENT_CLASSIFIER_PATH

logger = logging.getLogger("nova.intent_classifier")

try:
    import joblib
    from sentence_transformers import SentenceTransformer
    _DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - lets the server boot before deps are installed
    _DEPS_AVAILABLE = False


class IntentClassifier:
    def __init__(self):
        self._embedder = None
        self._clf = None
        self._classes = None
        self._loaded = False
        self._available = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if not _DEPS_AVAILABLE:
            logger.warning(
                "sentence-transformers/joblib not installed - trained intent "
                "classifier unavailable, all turns will use the LLM classification fallback."
            )
            return
        if not INTENT_CLASSIFIER_PATH.exists():
            logger.warning(
                "No trained intent classifier found at %s - all turns will use "
                "the LLM classification fallback. Run classifier/train_intent_classifier.py "
                "and copy the .joblib file into place to enable it.",
                INTENT_CLASSIFIER_PATH,
            )
            return

        try:
            bundle = joblib.load(INTENT_CLASSIFIER_PATH)
            self._embedder = SentenceTransformer(bundle["embedding_model_name"])
            self._clf = bundle["classifier"]
            self._classes = list(bundle.get("classes", getattr(self._clf, "classes_", [])))
            self._available = True
            logger.info(
                "Loaded trained intent classifier from %s (classes=%s)",
                INTENT_CLASSIFIER_PATH, self._classes,
            )
        except Exception:
            logger.exception(
                "Failed to load trained intent classifier - all turns will use "
                "the LLM classification fallback."
            )
            self._available = False

    @property
    def available(self) -> bool:
        self.load()
        return self._available

    def classify(self, text: str) -> Optional[dict]:
        """
        Returns {"intent": str, "confidence": float, "all_scores": dict}
        or None if the classifier is unavailable / inference fails -
        callers should treat None as "use the fallback classifier".
        """
        self.load()
        if not self._available:
            return None

        text = (text or "").strip()
        if not text:
            return None

        try:
            embedding = self._embedder.encode([text], normalize_embeddings=True)
            probs = self._clf.predict_proba(embedding)[0]
            classes = self._clf.classes_
            best_idx = probs.argmax()
            confidence = float(probs[best_idx])
            intent = str(classes[best_idx])
            all_scores = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
            return {"intent": intent, "confidence": confidence, "all_scores": all_scores}
        except Exception:
            logger.exception("Intent classifier inference failed for text=%r", text)
            return None


intent_classifier = IntentClassifier()