#!/usr/bin/env bash
# package.sh — Create a submission ZIP for upload
# Usage: bash package.sh [team_id]
# Example: bash package.sh TEAM42
#
# The resulting ZIP must contain:
#   manifest.json   ← team metadata (required)
#   agent.py        ← entry point declared in manifest["entry_point"]
#   *.py            ← any helper modules imported by agent.py
#   requirements.txt (optional, but recommended)
#
# DO NOT include:
#   dataset.csv, labels.npy, eval_features.csv, eval_labels.npy
#   __pycache__/, .git/, *.ipynb checkpoints

set -e

TEAM_ID="${1:-SAMPLE001}"
OUTFILE="${TEAM_ID}_submission.zip"

# Confirm we're in the submission directory
if [ ! -f "manifest.json" ] || [ ! -f "agent.py" ]; then
  echo "ERROR: Run this script from the directory containing manifest.json and agent.py"
  exit 1
fi

# Validate manifest has required fields
python3 - <<'PYEOF'
import json, sys
with open("manifest.json") as f:
    m = json.load(f)
required = ["team_name", "team_id", "institution", "members", "entry_point"]
missing = [k for k in required if k not in m]
if missing:
    print(f"ERROR: manifest.json missing fields: {missing}")
    sys.exit(1)
ep = m["entry_point"]
import os
if not os.path.exists(ep):
    print(f"ERROR: entry_point '{ep}' not found in current directory")
    sys.exit(1)
print(f"manifest.json OK — entry_point={ep}, team_id={m['team_id']}")
PYEOF

# Build the ZIP (exclude data files, caches, notebooks)
zip -r "${OUTFILE}" . \
  --exclude "*.csv" \
  --exclude "*.npy" \
  --exclude "__pycache__/*" \
  --exclude "*.pyc" \
  --exclude ".git/*" \
  --exclude "*.ipynb_checkpoints/*" \
  --exclude "*.zip" \
  --exclude "package.sh"

echo ""
echo "Created: ${OUTFILE}"
echo "Contents:"
unzip -l "${OUTFILE}"
