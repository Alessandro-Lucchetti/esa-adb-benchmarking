# tsadbench - common dev tasks. The conda env name matches the package.
ENV := tsadbench
RUN := conda run -n $(ENV)
CONFIG ?= configs/experiments/mission1_full_phase5.yaml

.DEFAULT_GOAL := help
.PHONY: help env env-update install test test-all golden run benchmark clean env-remove

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

env:  ## Create the tsadbench conda env and install the package (editable)
	conda env create -f setup/environment.yml
	$(RUN) python -m pip install -e ".[dev,taxonomy]"

env-update:  ## Update the env from environment.yml (prune removed deps)
	conda env update -f setup/environment.yml --prune

install:  ## (Re)install the package editable into the existing env
	$(RUN) python -m pip install -e ".[dev,taxonomy]"

test:  ## Fast tests (excludes the slow full-phase golden sweep)
	$(RUN) python -m pytest -m "not slow" -q

test-all:  ## All tests including the slow golden sweep across phases
	$(RUN) python -m pytest -q

golden:  ## Verify SMED labels are bit-exact vs the golden baseline (light phase 5)
	$(RUN) python -m pytest tests/test_golden_checksums.py::test_smed_light_phase5_bitexact -q

run:  ## Run an experiment: make run CONFIG=configs/experiments/mission1_light_phase5.yaml
	$(RUN) python scripts/run.py $(CONFIG)

benchmark:  ## Build and run the standalone SMED timing benchmark (Docker; run smed-benchmark/prepare_data.sh first)
	cd smed-benchmark && ./run.sh

clean:  ## Remove Python caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache

env-remove:  ## Delete the tsadbench conda env
	conda env remove -n $(ENV) -y
