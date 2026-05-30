# docs/baseline/

The **honest hardware baseline**, captured by `make bringup` **on the Jetson**.
These artifacts are committed as the record of what the device actually is and
what the camera actually does — measured, not asserted.

| File | Written by | What it proves |
|------|------------|----------------|
| `device_report.json` | `scripts/check_device.py` | Jetson model, JetPack/L4T, CUDA, TensorRT, GPU, power mode, camera nodes, readiness |
| `camera_report.json` | `scripts/camera_probe.py` | the GStreamer pipeline used + **measured** capture FPS, resolution, frame counts |
| `vigil_baseline_*.jpg` | `scripts/camera_probe.py` | real stills from the CSI sensor |

> Reports generated **off-device** (`on_target_hardware: false`) are not a
> baseline — they only confirm the scripts degrade cleanly. The committed
> baseline must come from a run on the Jetson. Regenerate with `make bringup`.
