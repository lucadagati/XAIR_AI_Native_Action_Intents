#!/usr/bin/env bash
# Download VisA (Visual Anomaly) industrial dataset — CC BY 4.0
# https://registry.opendata.aws/visa/
set -euo pipefail
RAW="$(cd "$(dirname "$0")/../../../.." && pwd)/experiments/datasets/raw"
mkdir -p "$RAW"
TAR="$RAW/VisA_20220922.tar"
URL="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"

if [ ! -f "$TAR" ]; then
  echo "Downloading VisA (~2 GB)..."
  wget -O "$TAR" "$URL" || curl -L -o "$TAR" "$URL"
fi

if [ ! -d "$RAW/candle" ] && [ ! -d "$RAW/VisA" ]; then
  echo "Extracting VisA..."
  tar -xf "$TAR" -C "$RAW"
fi

if [ -d "$RAW/VisA" ]; then
  echo "VisA ready at $RAW/VisA"
else
  echo "VisA ready at $RAW (category folders)"
fi
