#!/usr/bin/env bash
# scripts/train_classifier.sh
#
# Trains NOVA's intent classifier. Runs the actual work on WSL's native
# filesystem (~/nova-classifier) instead of /mnt/c/... because pip
# installing sentence-transformers (which pulls in PyTorch) is extremely
# slow across the Windows/Linux filesystem boundary. Source files and
# outputs live in v2/classifier/ as usual - this script just syncs to/
# from a fast scratch dir for the actual venv + training work.
set -euo pipefail

# Resolve v2/classifier/ relative to this script's location, so it works
# no matter which directory you run it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFIER_DIR="$(cd "$SCRIPT_DIR/../classifier" && pwd)"
WORK_DIR="$HOME/nova-classifier"

echo "== Source (Windows-mounted): $CLASSIFIER_DIR"
echo "== Work dir (native WSL fs): $WORK_DIR"

mkdir -p "$WORK_DIR"

echo "== Syncing source files to native filesystem ..."
cp "$CLASSIFIER_DIR"/*.py "$WORK_DIR"/
cp "$CLASSIFIER_DIR"/*.jsonl "$WORK_DIR"/
cp "$CLASSIFIER_DIR"/*.txt "$WORK_DIR"/

cd "$WORK_DIR"

if [ ! -d ".venv" ]; then
    echo "== No venv found - creating one ..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "== Installing/checking dependencies ..."
pip install -q -r requirements.txt

echo "== Training ..."
python train_intent_classifier.py

echo "== Copying results back to $CLASSIFIER_DIR ..."
cp intent_classifier.joblib "$CLASSIFIER_DIR"/
cp training_report.txt "$CLASSIFIER_DIR"/

echo ""
echo "== Done. training_report.txt and intent_classifier.joblib are in:"
echo "   $CLASSIFIER_DIR"