#!/usr/bin/env bash
# Create the tsadbench conda env and install the package (editable). Run from repo root.
set -euo pipefail
conda env create -f ./setup/environment.yml
conda run -n tsadbench python -m pip install -e ".[dev,taxonomy]"
echo "Done. Activate with:  conda activate tsadbench"
