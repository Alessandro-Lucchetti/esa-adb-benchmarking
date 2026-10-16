#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d data/ESA-Mission1/raw ]]; then
    echo "No data/ESA-Mission1/raw here: run ./prepare_data.sh first" >&2
    exit 1
fi

mkdir -p output

echo "=== Building Docker image (first build may take a few minutes) ==="
docker build -t smed-benchmark .

echo ""
echo "=== Running benchmark (single core, expect 30-60 min for full set) ==="
docker run --rm \
    --cpuset-cpus="0" \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd)/output:/output" \
    smed-benchmark

echo ""
echo "=== Done. Results: output/benchmark_results.csv ==="
