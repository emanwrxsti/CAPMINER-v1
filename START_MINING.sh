#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

POOL="${POOL:-stratum+tcp://us.icminers.com:7182}"
WALLET="${WALLET:-YOUR_ALPHA_WALLET}"
WORKER="${WORKER:-linux-rig}"
DEVICES="${DEVICES:-0}"
THREADS="${THREADS:-256}"
BLOCKS_PER_SM="${BLOCKS_PER_SM:-24}"
BATCH_MS="${BATCH_MS:-100}"
SUBMIT_FORMAT="${SUBMIT_FORMAT:-compact-decstr}"

if [[ "$DEVICES" == *,* ]]; then
  echo "Use one CapMiner process per GPU. DEVICES must contain one index." >&2
  echo "Example: DEVICES=0 WORKER=gpu0 ./START_MINING.sh" >&2
  exit 1
fi

if [[ "$WALLET" == "YOUR_ALPHA_WALLET" || -z "$WALLET" ]]; then
  echo "Set WALLET before starting, for example:" >&2
  echo "  WALLET=your_wallet DEVICES=0 ./START_MINING.sh" >&2
  exit 1
fi

exec ./capminer \
  --algo alphanumeric \
  --pool "$POOL" \
  --wallet "$WALLET" \
  --worker "$WORKER" \
  --pass x \
  --devices "$DEVICES" \
  --threads "$THREADS" \
  --blocks-per-sm "$BLOCKS_PER_SM" \
  --batch-ms "$BATCH_MS" \
  --alpha-submit-format "$SUBMIT_FORMAT" \
  --no-opencl \
  --log-file alpha-linux.log
