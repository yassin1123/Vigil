# Vigil — Jetson Orin Nano Bring-Up Runbook

This is a **runbook**, not prose: follow the steps in order, run the exact
commands, and compare against the expected output. By the end you will have a
flashed Jetson Orin Nano, a confirmed CSI camera, and a committed baseline
(`docs/baseline/device_report.json` + camera stills + measured capture FPS).

Everything here was written so a stranger with the same hardware can reproduce
it. Where a value depends on your JetPack point-release, that is called out
rather than hard-coded.

---

## 0. What you need

| Item | Notes |
|------|-------|
| Jetson Orin Nano Developer Kit | The board in hand. |
| CSI camera module | IMX219-class (e.g. Raspberry Pi Camera v2 / equivalent). |
| microSD card | 64 GB+ UHS-1, **or** an NVMe SSD if booting from NVMe. |
| Host PC | For flashing. Ubuntu 20.04/22.04 host for SDK Manager; any OS for the SD-card image method. |
| USB-C / barrel power | Use the recommended PSU — under-powered supplies cause Argus/CUDA flakiness. |
| Ethernet + keyboard/monitor | For first boot (headless setup is possible but out of scope here). |

> **Network note:** you need the network **only** to flash and install packages
> during bring-up. Vigil itself runs fully offline; Day 7 proves it. Do not bake
> any runtime network dependency into anything you install here.

---

## 1. Flash JetPack 6.x

Pick **one** method.

### Method A — SD-card image (simplest)

1. Download the **Jetson Orin Nano Developer Kit SD Card Image** for JetPack 6.x
   from the NVIDIA Jetson Download Center.
2. Flash it with Balena Etcher (or `dd`):

   ```bash
   # Linux/macOS, replace /dev/sdX with your card (double-check!)
   sudo dd if=jp6x-orin-nano-sd-card-image.img of=/dev/sdX bs=1M status=progress conv=fsync
   sync
   ```

3. Insert the card, connect monitor/keyboard/network, power on, and complete the
   first-boot wizard (user, locale, network).

### Method B — SDK Manager (NVMe / full control)

1. On an Ubuntu host, install **NVIDIA SDK Manager**.
2. Put the Orin Nano into **Force Recovery** mode (jumper FC REC + GND, then
   power on) and connect it to the host over USB-C.
3. In SDK Manager: select **Jetson Orin Nano**, **JetPack 6.x**, and flash. Let
   it install the full runtime (CUDA, cuDNN, TensorRT).

Either way, after first boot you should be at a desktop/terminal on the Jetson.

---

## 2. Update and confirm the OS

```bash
sudo apt-get update && sudo apt-get upgrade -y
cat /etc/nv_tegra_release
```

**Expected** (exact revision will vary):

```
# R36 (release), REVISION: 3.0, GCID: 00000000, BOARD: generic, EABI: aarch64, ...
```

`R36` ⇒ JetPack 6.x. If you see `R35`, you are on JetPack 5.x — reflash with a
JetPack 6.x image.

---

## 3. Set the power mode and lock clocks

Vigil's baseline must state the power envelope it was measured under, so set it
explicitly and record it.

```bash
# Show available power modes and the current one
sudo nvpmodel -q

# Set maximum performance (mode 0 = MAXN on Orin Nano)
sudo nvpmodel -m 0

# Lock clocks to max (optional, for repeatable benchmarks)
sudo jetson_clocks

# Confirm
sudo nvpmodel -q
```

**Expected** (mode name varies by JetPack):

```
NV Power Mode: MAXN
0
```

> For honest power numbers later (Day 6/7), you may instead pin a **fixed** mode
> (e.g. 15W) and report that. The rule is: whatever mode you measure under, it
> is recorded in the baseline and the final report.

---

## 4. Verify CUDA + TensorRT

```bash
# CUDA
cat /usr/local/cuda/version.json
nvcc --version            # if nvcc isn't found, add CUDA to PATH (see below)

# TensorRT (Python runtime — this is what Vigil uses)
python3 -c "import tensorrt as trt; print('TensorRT', trt.__version__)"

# TensorRT (packages)
dpkg -l | grep -i tensorrt
```

**Expected** (versions vary with JetPack point-release):

```
{ "cuda" : { "name" : "CUDA SDK", "version" : "12.2.140" } }
...
TensorRT 8.6.2          # (example)
```

If `nvcc` is missing, add CUDA to your PATH and re-login:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 5. Connect and verify the CSI camera

1. Power **off** the Jetson before connecting the ribbon.
2. Seat the CSI ribbon in connector **CAM0** — contacts facing the heatsink/SoM,
   gold fingers toward the board, latch closed.
3. Power on, then:

```bash
# Camera should enumerate as a video node
ls /dev/video*

# Argus smoke test (no display needed) — should reach PLAYING then EOS cleanly
gst-launch-1.0 nvarguscamerasrc num-buffers=1 ! fakesink -v
```

**Expected:** at least one `/dev/video0`, and the `gst-launch` command prints
sensor mode lines (resolution/framerate) and exits without error.

If `/dev/video*` is empty: re-seat the ribbon, confirm it's in CAM0, reboot. A
backwards or half-latched ribbon is the #1 cause.

---

## 6. Get Vigil and run the baseline

```bash
git clone <vigil-repo-url> vigil
cd vigil

make bringup
```

`make bringup` runs both probes and writes artifacts to `docs/baseline/`:

| Artifact | From | Contents |
|----------|------|----------|
| `device_report.json` | `check_device.py` | model, JetPack/L4T, CUDA, TensorRT, GPU, power mode, camera nodes, readiness flag |
| `camera_report.json` | `camera_probe.py` | pipeline used, measured FPS, resolution, frame counts |
| `vigil_baseline_*.jpg` | `camera_probe.py` | real stills from the sensor |

**Expected console (abridged):**

```
  [OK ] model            NVIDIA Jetson Orin Nano Developer Kit
  [OK ] jetpack/l4t      JetPack 6.x  (L4T R36.3.0)
  [OK ] cuda             12.2.140
  [OK ] tensorrt         8.6.2
  [OK ] gpu              Tegra integrated GPU present
  [OK ] power_mode       MAXN
  [OK ] camera           1 video node(s): /dev/video0
  READY: CUDA, TensorRT, GPU and a camera node are all present.
  ...
  MEASURED FPS    : 29.7
```

> The FPS you record is **measured**, not promised. Whatever it is (e.g. 14 FPS
> at 1080p, 60 FPS at 720p), that is the number that goes in the baseline.

### Tuning the camera probe

```bash
# 720p high-framerate baseline
python3 scripts/camera_probe.py --width 1280 --height 720 --framerate 60

# Flip 180° if the module is mounted upside-down
python3 scripts/camera_probe.py --flip-method 2

# Second camera on a dual-CSI carrier
python3 scripts/camera_probe.py --sensor-id 1
```

Documented IMX219 sensor modes: `3280x2464@21`, `1920x1080@30`, `1640x1232@30`,
`1280x720@60`, `1280x720@120`.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ls /dev/video*` empty | Ribbon backwards / not latched | Power off, re-seat in CAM0 contacts-to-heatsink, reboot |
| `nvarguscamerasrc` errors / no frames | Wrong sensor mode | Use a documented mode (§6); check `--sensor-id` |
| `Could not open the camera` from probe | No GStreamer in OpenCV, or no camera | On dev PC this is expected; on Jetson use system OpenCV |
| `nvcc: command not found` | CUDA not on PATH | §4 PATH export |
| `import tensorrt` fails | Wrong Python / TRT not installed | Use system `python3`; `sudo apt-get install nvidia-tensorrt` |
| Argus flaky, random failures | Under-powered PSU | Use the recommended supply; avoid weak USB-C bricks |
| `nvpmodel: command not found` | Not on Jetson, or PATH | Expected off-device; on Jetson it's in `/usr/sbin` |

---

## 8. Off-device behaviour (why CI stays green)

Both scripts run on any machine. Off a Jetson they print a clear **"NOT ON
TARGET HARDWARE"** message, write a report that records the absence, and exit
`0`. This is intentional: per Vigil's both-worlds rule, nothing in the repo may
*require* the GPU to run, so the bring-up scripts are safe to execute (and to
keep in CI) without a Jetson attached.

What you **cannot** fake off-device: the measured camera FPS and the populated
`device_report.json`. Those only become real on the hardware — which is the
whole point of this session.
