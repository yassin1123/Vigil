# calibration/

INT8 calibration images for `scripts/build_engine.py`. TensorRT runs the network
on these to choose INT8 quantisation scales, so they should resemble the scene
the camera will actually watch.

`images/` ships with a **small synthetic placeholder set** so an INT8 build works
out of the box. They are deterministic geometric shapes — enough to exercise the
calibration path, **not** representative of any real scene.

## For real INT8 accuracy

Replace the placeholders with 50–500 frames from your actual deployment:

```bash
# After bring-up, the baseline stills are a good starting point:
cp docs/baseline/vigil_baseline_*.jpg calibration/images/
# ...or capture a representative clip and drop the frames in here.
```

Then rebuild:

```bash
python3 scripts/build_engine.py --weights yolov8n.pt --precision int8
```

If `images/` is empty, the build falls back to **FP16** with a printed note —
INT8 is never silently skipped.
