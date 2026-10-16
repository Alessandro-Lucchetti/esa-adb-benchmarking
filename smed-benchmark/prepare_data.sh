#!/usr/bin/env bash
# Copy the ESA-Mission1 inputs the benchmark needs into ./data (data only, nothing else).
#   ./prepare_data.sh [DATA_DIR]
# DATA_DIR is the repository data folder holding metadata.json and ESA-Mission1/{labels.csv,raw/}
# (default: ../data, or $SMED_DATA). The raw parquets come from setup/convert_zips_to_parquet.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:-${SMED_DATA:-$SCRIPT_DIR/../data}}"
MISSION="ESA-Mission1"

if [[ ! -d "$SRC/$MISSION/raw" ]]; then
    echo "No $SRC/$MISSION/raw folder: convert the ESA-ADB zips first (setup/convert_zips_to_parquet.py)" >&2
    exit 1
fi

echo "=== Copying data files from $SRC ==="
mkdir -p "$SCRIPT_DIR/data/$MISSION/raw"
cp "$SRC/metadata.json" "$SCRIPT_DIR/data/metadata.json"
cp "$SRC/$MISSION/labels.csv" "$SCRIPT_DIR/data/$MISSION/labels.csv"
cp "$SRC/$MISSION/raw"/channel_*.parquet "$SCRIPT_DIR/data/$MISSION/raw/"
echo "  Done: metadata.json, labels.csv, $(ls "$SCRIPT_DIR/data/$MISSION/raw/" | wc -l) channel files"
