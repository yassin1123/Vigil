# models/

Built model artifacts live here: `yolov8n.onnx`, `yolov8n.engine`, and the
`*.engine.json` build record produced by `scripts/build_engine.py`.

**Nothing in this directory is committed except this README.** A TensorRT
`.engine` is hardware- and version-specific — it is tuned to the exact GPU,
TensorRT, and CUDA on the build machine and will not load anywhere else. Build
it on the Jetson:

```bash
python3 scripts/build_engine.py --weights yolov8n.pt
# -> models/yolov8n.onnx, models/yolov8n.engine, models/yolov8n.engine.json
```

The sidecar `yolov8n.engine.json` records precision (int8/fp16/fp32), workspace,
input size, TensorRT version, and the device model — your honest build receipt.
