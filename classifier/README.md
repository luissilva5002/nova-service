# NOVA intent classifier - training folder

Trains a small, non-generative intent classifier (MiniLM sentence
embeddings + LogisticRegression) to replace the LLM-based
`classify_intent_llm()` call, with real calibrated confidence scores
instead of a model self-reporting a confidence number.

## Files

- `intent_dataset.jsonl` - seed labeled examples (~140) across all 6
  NOVA intents (`action_music`, `action_calendar`, `action_code`,
  `memory_write`, `memory_recall`, `chat`), including the specific
  novel phrasings from the logged misclassifications ("look into
  those files", "the record you have") and hard negatives (chat
  sentences that mention memory/music words without being that
  intent).
- `train_intent_classifier.py` - encodes the dataset, trains the
  classifier, prints a classification report, dumps every wrong
  validation prediction with its confidence, and sweeps confidence
  thresholds so you can pick one with real numbers instead of
  guessing.
- `classify_test.py` - loads the trained model and lets you type
  sentences to sanity-check predictions before wiring it into NOVA.
- `requirements.txt` - pinned dependencies.

## Setup

```bash
cd intent_classifier_training
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First run of the training script downloads `all-MiniLM-L6-v2`
(~90MB) from Hugging Face and caches it locally - after that it's
fully offline.

## Train

```bash
python train_intent_classifier.py
```

This writes two files back into this folder:
- `intent_classifier.joblib` - the trained model (embedder name +
  sklearn classifier bundled together)
- `training_report.txt` - classification report, every wrong
  validation prediction with its confidence, and the threshold
  sweep table

**Read `training_report.txt` before doing anything else.** The part
that matters most is the "Wrong predictions" section - a wrong
prediction should show a *low* confidence number. If you see a wrong
prediction with confidence above ~0.8, that specific phrasing needs
more/better examples in `intent_dataset.jsonl` before you can trust
the threshold to catch it; add 3-5 similar variants and retrain.

## Pick a threshold

Look at the threshold sweep table in `training_report.txt`:

```
 threshold   kept  forced_to_chat  acc_on_kept
       0.3     28               0        0.893
       0.5     26               2        0.923
       0.7     22               6        0.955
       0.9     15              13        1.000
```

Pick the lowest threshold where `acc_on_kept` is high enough for you
(0.95+ is a reasonable bar for `memory_recall`, since that's the
category where a wrong-but-confident call leads to the fabrication
bug). Set that number as `CONFIDENCE_THRESHOLD` in `classify_test.py`
for testing, and later as `INTENT_CONFIDENCE_THRESHOLD` in NOVA's
`config.py`.

## Test it manually

```bash
python classify_test.py "check your memory for the pcb file"
python classify_test.py "play some jazz"
python classify_test.py "I was thinking about my old files today"

# or interactively:
python classify_test.py
```

## Retraining after corrections (ongoing loop)

Once this is wired into NOVA and logging turns (`data/intent_log.jsonl`
as discussed), the loop is:

1. Pull rows where confidence was low, or where the classifier's
   `raw_intent` disagreed with what actually should have happened.
2. Hand-correct the label, append as a new line to
   `intent_dataset.jsonl` in this same JSONL shape:
   `{"text": "...", "intent": "..."}`.
3. Re-run `python train_intent_classifier.py`. Takes seconds - no
   GPU, no special infra.
4. Copy the new `intent_classifier.joblib` to wherever NOVA loads it
   from (see `INTENT_CLASSIFIER_PATH` in `config.py` once that's
   wired up).

## Next step

Once you're happy with the threshold and the wrong-prediction list is
clean, say the word and I'll wire `intent_classifier.py` into NOVA's
actual pipeline (`config.py`, `main.py`, `intent_router.py`) using
this trained artifact.
