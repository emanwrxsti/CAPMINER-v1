#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-linux}"
OUT_DIR="${OUT_DIR:-$ROOT/dist/linux}"
VERSION="${VERSION:-1.0.0}"

[[ -x "$BUILD_DIR/capminer" ]] || {
  echo "Missing $BUILD_DIR/capminer. Run scripts/build-linux.sh first." >&2
  exit 1
}

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp "$BUILD_DIR/capminer" "$OUT_DIR/"
cp "$ROOT/START_MINING.sh" "$OUT_DIR/"
cp "$ROOT/README-LINUX.md" "$OUT_DIR/"
chmod +x "$OUT_DIR/capminer" "$OUT_DIR/START_MINING.sh"
(
  cd "$OUT_DIR"
  sha256sum capminer START_MINING.sh README-LINUX.md > SHA256SUMS
)

tar -C "$(dirname "$OUT_DIR")" -czf "$ROOT/dist/capminer-linux-x64-v${VERSION}.tar.gz" "$(basename "$OUT_DIR")"
sha256sum "$ROOT/dist/capminer-linux-x64-v${VERSION}.tar.gz" > "$ROOT/dist/capminer-linux-x64-v${VERSION}.tar.gz.sha256"
echo "Created $ROOT/dist/capminer-linux-x64-v${VERSION}.tar.gz"
