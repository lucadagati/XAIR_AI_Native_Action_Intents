#!/usr/bin/env bash
# Download MVTec AD (CC BY-NC-SA 4.0, academic use only).
# Prefer classic single-archive mirror (avoids HF multi-file 429 rate limits).
# Cite Bergmann et al. / MVTec AD license.
set -euo pipefail
RAW="$(cd "$(dirname "$0")/../../../.." && pwd)/experiments/datasets/raw"
DEST="$RAW/mvtec_ad"
ARCHIVE="$RAW/mvtec_anomaly_detection.tar.xz"
# Ungated HF mirror with classic MVTec folder layout (15 categories)
URL="${MVTEC_AD_URL:-https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/main/mvtec_anomaly_detection.tar.xz}"

mkdir -p "$RAW"

if [ -f "$DEST/.complete" ] && [ -d "$DEST/bottle/test/good" ]; then
  echo "MVTec AD already complete at $DEST"
  find "$DEST" -name '*.png' | wc -l
  exit 0
fi

if [ ! -f "$ARCHIVE" ] || [ "$(stat -c%s "$ARCHIVE" 2>/dev/null || echo 0)" -lt 1000000000 ]; then
  echo "Downloading classic MVTec AD archive (~5 GB)..."
  wget -c --progress=dot:giga -O "$ARCHIVE" "$URL" || curl -L -C - -o "$ARCHIVE" "$URL"
fi

echo "Extracting..."
TMP="$RAW/mvtec_ad_extract_tmp"
rm -rf "$TMP"
mkdir -p "$TMP"
tar -xJf "$ARCHIVE" -C "$TMP"

# Normalize to DEST with classic layout (bottle/, cable/, ...)
rm -rf "$DEST"
if [ -d "$TMP/bottle" ]; then
  mv "$TMP" "$DEST"
elif [ -d "$TMP/mvtec_anomaly_detection/bottle" ]; then
  mv "$TMP/mvtec_anomaly_detection" "$DEST"
  rm -rf "$TMP"
else
  BOT=$(find "$TMP" -maxdepth 3 -type d -name bottle | head -1)
  if [ -z "$BOT" ]; then
    echo "Could not find classic MVTec layout after extract" >&2
    exit 1
  fi
  mv "$(dirname "$BOT")" "$DEST"
  rm -rf "$TMP"
fi

touch "$DEST/.complete"
echo "MVTec AD ready at $DEST"
find "$DEST" -name '*.png' | wc -l
ls "$DEST"
