# Vigil Log Export — Roadmap (built vs. rails)

Vigil's evidentiary log (Day 5) is written and verified on-device. Getting that
record **off** the device is done through one seam — the `ExportSink` interface
in [src/vigil/log/sinks.py](../src/vigil/log/sinks.py). One backend is built
today; the rest are deliberately **rails only**: interfaces + honest stubs, so
the roadmap is credible without faking radios or crypto in the core.

> **The honesty rule for this file:** if it's listed as a stub, no real transport
> or crypto exists. The stub's `handle()`/`export()` return `implemented=False`
> (never a silent success), and its concrete method (`stream_event` / `burst` /
> `write`) raises `NotImplementedError`. A reviewer can `grep implemented=False`
> and trust the boundary.

## Status

| Backend | Class | Trigger | Mode | Status | Emits |
|---|---|---|---|---|---|
| USB removable drive | `USBExportSink` | drive inserted | batch (whole log + re-verify) | **BUILT** (Day 5) | `LOG_EXPORTED` |
| Local directory | `LocalDirSink` | drive inserted | batch | **BUILT** (CI/test stand-in) | `LOG_EXPORTED` |
| Disabled | `NullSink` | — | — | **BUILT** (no-op) | — |
| Mesh stream | `MeshStreamSink` | peer available / event appended | stream (per event) | **RAILS ONLY** | `MESH_PEER_ACK` |
| RF burst | `RFBurstSink` | signal loss | burst (last N entries) | **RAILS ONLY** | `RF_BURST_SENT` |
| Armoured module | `ArmoredModuleSink` | event appended | stream (per event) | **RAILS ONLY** | `ARMORED_WRITE` |

## How routing works (so a new backend is "implement an interface, not edit core")

- Each sink declares a `SinkCapabilities` (triggers, mode, `implemented`).
- `ExportRouter.dispatch(DispatchContext(trigger, ...))` sends the opportunity to
  every sink whose declared triggers include that trigger — it never checks the
  sink's *type*.
- The app detects triggers (a drive appearing, a mesh peer answering, telemetry
  reporting signal loss, a new event being logged) and calls `dispatch`.
- Adding a real backend later = subclass `ExportSink`, fill in `capabilities()`
  and the transport method. No router or logger change.

## Mapping to the product brief phases

The brief scopes the 8-day build to the **inference core** and explicitly lists
mesh networking, RF burst, and the armoured memory module as **out of scope /
forward-looking**. This session implements the brief's one forward-looking
concession — the export abstraction — as rails:

- **Phase 1 (this build): USB export, built and verified.** A copy is made to a
  removable drive and the chain is re-verified *at the destination*. Acknowledged
  limitation: if the device is destroyed before any export, the on-device log is
  lost — USB alone is not a black box.
- **Phase 2 (roadmap): mesh streaming.** Push each event to peer units over a
  local link so the record survives loss of any single unit. Rail: `MeshStreamSink`.
- **Phase 3 (roadmap): RF burst.** On a low-power / low-signal condition, transmit
  the last N entries off-site. Rail: `RFBurstSink`.
- **Phase 4 (roadmap): armoured memory.** A separable, encrypted store that
  survives device destruction — the actual black box. Rail: `ArmoredModuleSink`.

## Known gap carried forward

A bare hash chain detects edit/insert/delete/reorder but **not tail truncation**
(a valid prefix survives). The mitigation — periodically anchoring/signing the
terminal hash, ideally pushed off-device by one of the backends above — is a
known follow-up, noted in [verify.py](../src/vigil/log/verify.py) and flagged for
the airgap report.
