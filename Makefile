# Vigil — Edge Inference Core
#
# Dev targets (`test`, `lint`, `run`) work on any machine with mocked hardware.
# Bring-up targets (`bringup`, ...) write the on-device baseline on the Jetson.
# Everything degrades gracefully off-device.

PYTHON   ?= python3
SRC      := src
SCRIPTS  := scripts
BASELINE := docs/baseline
LOG      ?= logs/events.jsonl

.DEFAULT_GOAL := help

.PHONY: help install test lint format run \
        bringup check-device camera-probe clean-baseline clean

help: ## Show available targets
	@echo "Vigil — make targets:"
	@echo "  make install       Editable install with dev extras (cv2, pytest, ruff)"
	@echo "  make test          Run the unit tests (no hardware required)"
	@echo "  make lint          Lint with ruff"
	@echo "  make format        Auto-format + autofix with ruff"
	@echo "  make run           Capture -> detect -> track over the configured source"
	@echo "  make track-demo    Print a track timeline for the committed demo clip"
	@echo "  make verify-log    Verify the hash-chained event log (LOG=path)"
	@echo "  make bringup       On-Jetson: device + camera baseline -> docs/baseline/"
	@echo "  make check-device  Probe Jetson/CUDA/TensorRT/camera"
	@echo "  make camera-probe  Capture CSI frames, measure FPS"
	@echo "  make clean         Remove caches"

install: ## Editable install with dev extras (dev/CI only — not on the Jetson)
	$(PYTHON) -m pip install -e ".[dev]"

test: ## Run unit tests (mocked hardware; never needs the GPU)
	$(PYTHON) -m pytest

lint: ## Lint sources, tests, and scripts
	$(PYTHON) -m ruff check $(SRC) tests $(SCRIPTS)

format: ## Auto-format and apply safe fixes
	$(PYTHON) -m ruff format $(SRC) tests $(SCRIPTS)
	$(PYTHON) -m ruff check --fix $(SRC) tests $(SCRIPTS)

run: ## Capture -> detect -> track over the configured source
	PYTHONPATH=$(SRC) $(PYTHON) -m vigil run --max-frames 100

track-demo: ## Print a track timeline for the committed demo clip (no GPU)
	PYTHONPATH=$(SRC) $(PYTHON) -m vigil track-demo

verify-log: ## Verify the hash-chained event log (LOG=path to override)
	PYTHONPATH=$(SRC) $(PYTHON) -m vigil log verify $(LOG)

bringup: check-device camera-probe ## Full Day-1 bring-up baseline (Jetson)
	@echo ""
	@echo "Bring-up complete. Baseline artifacts in $(BASELINE)/:"
	@ls -1 $(BASELINE) 2>/dev/null || echo "  (none — check the output above)"

check-device: ## Probe device readiness
	@mkdir -p $(BASELINE)
	$(PYTHON) $(SCRIPTS)/check_device.py --output $(BASELINE)/device_report.json

camera-probe: ## Probe CSI camera and measure capture FPS
	@mkdir -p $(BASELINE)
	$(PYTHON) $(SCRIPTS)/camera_probe.py --output-dir $(BASELINE)

clean-baseline: ## Remove generated baseline artifacts
	rm -f $(BASELINE)/device_report.json $(BASELINE)/camera_report.json
	rm -f $(BASELINE)/vigil_baseline_*.jpg

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
