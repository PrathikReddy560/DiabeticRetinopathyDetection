# Raspberry Pi Deployment — DR Screening (INT8, CPU-only)

Two-stage diabetic retinopathy screening pipeline, INT8-quantized, running
entirely on a Raspberry Pi 5 CPU (no GPU, no PyTorch on device):

1. **Stage 1 — GANomaly gate** (INT8 ONNX, 128×128): anomaly score → threshold
   gate at `0.6282`. Healthy images stop here.
2. **Stage 2 — EfficientNet-B0 + VBLL** (INT8 ONNX, 224×224): severity grade
   0–4 with Bayesian uncertainty. 30 posterior weight samples are drawn in
   **NumPy** (the backbone runs once), giving mean probabilities, confidence
   `(1 − std) × 100`, and a low-confidence flag (`std > 0.15` →
   **Refer for Manual Review**).

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | Exact training preprocessing (crop + CLAHE + resize) |
| `inference.py` | Pipeline class + CLI |
| `demo_app.py` | Flask web demo (laptop browser ↔ Pi) |
| `benchmark.py` | Per-stage latency report |
| `models/` | **You put the INT8 artifacts here** (see below) |

## 1. Get the model artifacts

Run the two export notebooks in `deploy/kaggle_export/` on Kaggle (they load
your trained checkpoints, quantize to INT8, and verify the parity gates).
Download from each notebook's Output → `deploy/` and place in `models/`:

```
models/ganomaly_int8.onnx        (from export_stage1_ganomaly.ipynb)
models/stage1_deploy.json        (from export_stage1_ganomaly.ipynb)
models/stage2_vbll_int8.onnx     (from export_stage2_vbll.ipynb)
models/vbll_head.npz             (from export_stage2_vbll.ipynb)
models/stage2_deploy.json        (from export_stage2_vbll.ipynb)
```

## 2. Setup (Pi or laptop rehearsal — identical)

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements_pi.txt
```

## 3. Run

**Single image (CLI):**
```bash
python inference.py /path/to/fundus.jpg --models models
python inference.py /path/to/fundus.jpg --models models --force   # grade even if gate says normal
```

**Web demo:**
```bash
python demo_app.py --models models --host 0.0.0.0 --port 5000
# laptop browser -> http://<pi-ip>:5000
```

**Benchmark:**
```bash
python benchmark.py --models models --images /path/to/test/images --limit 50
```

## 4. Expected latency (RPi 5, INT8, 4 threads)

| Stage | Estimate |
|---|---|
| preprocess | ~3–8 ms |
| Stage 1 (GANomaly 128²) | ~25–60 ms |
| Stage 2 (B0 224², only on flagged) | ~40–80 ms |
| VBLL head (30 NumPy samples) | <2 ms |
| **Combined worst case** | **~80–150 ms** (< 200 ms budget) |

If you are over budget: reduce `--threads` contention is rarely the issue on
Pi 5; check nothing else runs; worst case, run the gate-only mode (Stage 1
only) for screening duty.

## 5. Deliberate on-device limitations (by design)

- **TTA is OFF** on device (4× backbone cost would bust the latency budget).
  Kaggle metrics include TTA; expect ~1–2% accuracy difference — that is the
  measured cost of edge deployment, report it honestly.
- **Grad-CAM is not available on device** (needs gradients; INT8 models are
  inference-only). The on-device explanation is the anomaly score, the grade
  probabilities, and the confidence/flag. Grad-CAM lives in the Kaggle
  notebooks for the thesis.
- The lesion-evidence fusion audit (Grad-CAM based) likewise stays on Kaggle;
  the Refer-for-Review decision on device is driven by the VBLL uncertainty.

## 6. Retraining on hospital data later

See `deploy/clinical/`. Convert the hospital export with
`prepare_clinical_dataset.py`, retrain on Kaggle (Run All), re-run the two
export notebooks, replace the files in `models/` — done.
