#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d data/ESA-Mission1/raw ]]; then
    echo "No data/ESA-Mission1/raw here: run ./prepare_data.sh first" >&2
    exit 1
fi

mkdir -p output

echo "=== Building Docker image (first build may take a few minutes) ==="
docker build -t ool-benchmark .

echo ""
echo "=== Running benchmark (single core, expect a few minutes for both sets) ==="
docker run --rm \
    --cpuset-cpus="0" \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd)/output:/output" \
    ool-benchmark

echo ""
echo "=== Done. Results: output/benchmark_results.csv ==="
