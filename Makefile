# Vigil — Edge Inference Core
# Day-1 bring-up targets. `make bringup` writes the baseline artifacts under
# docs/baseline/. Every target is safe to run off-device (the scripts degrade
# gracefully and exit cleanly when not on a Jetson).

PYTHON      ?= python3
SCRIPTS     := scripts
BASELINE    := docs/baseline

.DEFAULT_GOAL := help

.PHONY: help bringup check-device camera-probe clean-baseline

help: ## Show available targets
	@echo "Vigil — make targets:"
	@echo "  make bringup       Run device + camera probes, write docs/baseline/"
	@echo "  make check-device  Probe Jetson/CUDA/TensorRT/camera -> device_report.json"
	@echo "  make camera-probe  Capture CSI frames, measure FPS -> camera_report.json"
	@echo "  make clean-baseline  Remove generated baseline artifacts"

bringup: check-device camera-probe ## Full Day-1 bring-up baseline
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
