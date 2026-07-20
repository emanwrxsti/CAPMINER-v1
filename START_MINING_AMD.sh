#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

POOL="${POOL:-stratum+tcp://us.icminers.com:7182}"
WALLET="${WALLET:-YOUR_ALPHA_WALLET}"
WORKER="${WORKER:-amd-linux-rig}"
DEVICES="${DEVICES:-0}"
THREADS="${THREADS:-256}"
BLOCKS_PER_CU="${BLOCKS_PER_CU:-8}"
BATCH_MS="${BATCH_MS:-100}"
SUBMIT_FORMAT="${SUBMIT_FORMAT:-compact-decstr}"

if [[ "$WALLET" == "YOUR_ALPHA_WALLET" || -z "$WALLET" ]]; then
  echo "Set WALLET first, for example:" >&2
  echo "  WALLET=your_wallet ./START_MINING_AMD.sh" >&2
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
  --blocks-per-sm "$BLOCKS_PER_CU" \
  --batch-ms "$BATCH_MS" \
  --alpha-submit-format "$SUBMIT_FORMAT" \
  --no-opencl \
  --log-file alpha-amd-linux.log
