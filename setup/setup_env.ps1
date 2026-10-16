# Create the tsadbench conda env and install the package (editable). Run from repo root.
conda env create -f ./setup/environment.yml
conda run -n tsadbench python -m pip install -e ".[dev,taxonomy]"
conda activate tsadbench
