# RetinaAI — Edge-Deployable Autonomous Diabetic Retinopathy Screening System

[![Platform](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-red.svg)](https://www.raspberrypi.com/)
[![Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20%2B%20NumPy-blue.svg)](https://onnxruntime.ai/)
[![XAI](https://img.shields.io/badge/Explainability-Grad--CAM-emerald.svg)]()
[![Uncertainty](https://img.shields.io/badge/Uncertainty-VBLL%20%28Bayesian%29-amber.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, ultra-low-cost (**₹10,390 / ~$125**), point-of-care **Diabetic Retinopathy (DR) screening system** designed for resource-constrained clinical settings. The system couples a low-cost ophthalmic lens apparatus with a cascaded two-stage deep learning pipeline executing on a **Raspberry Pi 5 (4GB RAM)** in **under 200 ms per image** without cloud connectivity or GPU acceleration.

---

## 🌟 Key Innovations & Software Highlights

1. **Two-Stage Cascaded Edge Architecture**:
   - **Stage 1 (GANomaly Gate)**: One-class anomaly detection gate that screens healthy fundus images in **~30 ms**.
   - **Stage 2 (VBLL Severity Grader)**: EfficientNet-B0 + Variational Bayesian Last Layer (VBLL) grading DR severity across 5 classes (0–4).
2. **Epistemic Uncertainty Quantification**:
   - Uses **30 posterior weight samples** drawn from the learned Gaussian weight distribution $q(W) \sim \mathcal{N}(\mu_W, \sigma^2_W)$.
   - Automatically triggers a **"Refer for Manual Review"** safety flag when top-class predictive standard deviation exceeds $0.15$.
3. **Explainable AI (XAI) via Grad-CAM**:
   - Generates spatial feature maps ($1280 \times 7 \times 7$) combined with VBLL weights $\mu_W$ to render real-time **Grad-CAM heatmaps**.
   - Computes **Lesion Activation Load (%)** to clinically justify severity ratings (microaneurysms, hemorrhages, hard exudates).
4. **Framework-Free Edge Execution**:
   - **Zero PyTorch/TensorFlow on device**: Backbone feature extraction runs via **ONNX Runtime**, and Bayesian sampling runs in **pure NumPy**.
   - Total model footprint: **~30 MB FP32 ONNX**, peak RAM usage **< 870 MB**.
5. **Clinical Single-Page Web Application**:
   - Modern dark glassmorphism web UI (`demo_app.py`) built with Flask & JavaScript.
   - Live drag-and-drop fundus analysis, animated probability distribution charts, and side-by-side Grad-CAM heatmap visualization.

---

## 📐 System Architecture

```
                                  [ Fundus Image Input ]
                                             │
                                             ▼
                       ┌──────────────────────────────────────────┐
                       │  Preprocessing (Fundus Crop + CLAHE 128) │
                       └─────────────────────┬────────────────────┘
                                             │
                                             ▼
                       ┌──────────────────────────────────────────┐
                       │   Stage 1: GANomaly Gate (FP32 ONNX)     │
                       │   Anomaly Score vs Threshold (0.6282)    │
                       └─────────────┬────────────────┬───────────┘
                                     │                │
                      [ Score ≤ 0.6282 ]            [ Score > 0.6282 ]
                                     │                │
                                     ▼                ▼
                           ┌──────────────────┐  ┌──────────────────────────────────────────┐
                           │   NORMAL GATE    │  │  Stage 2: EfficientNet-B0 (224x224)    │
                           │   Stop (~30 ms)  │  │  + VBLL Head (30 Posterior Samples)      │
                           └──────────────────┘  └────────────────────┬─────────────────────┘
                                                                      │
                                                                      ▼
                                                         ┌───────────────────────────┐
                                                         │ Grad-CAM Heatmap & Lesion │
                                                         │ Load Evaluation (~75 ms)  │
                                                         └────────────┬──────────────┘
                                                                      │
                                                                      ▼
                                                         ┌───────────────────────────┐
                                                         │ Clinical Report & Referral│
                                                         │ Output (< 200 ms Total)   │
                                                         └───────────────────────────┘
```

---

## 📊 Model Performance Summary

| Model / Stage | Architecture | Target / Task | Metric | Value | Deployment Format |
|---|---|---|---|---|---|
| **Stage 1 Gate** | GANomaly Generator (EDE) | Anomaly Detection | **Test AUC** | **0.9093** | FP32 ONNX (`14.6 MB`) |
| **Stage 2 Grader** | EfficientNet-B0 + VBLL | 5-Class DR Grading | **Test Accuracy** | **0.8321** | FP32 ONNX (`16.0 MB`) |
| **Stage 2 Grader** | EfficientNet-B0 + VBLL | Quadratic Weighted Kappa | **Test QWK** | **0.8783** | `vbll_head.npz` (`52 KB`) |

> **Quantization Empirical Note**: Static INT8 quantization was evaluated and rejected. INT8 caused GANomaly AUC to collapse to `0.55` and VBLL QWK to drop to `0.17` due to fine-grained weight variance ($\sigma \approx 0.007$) being zeroed out. Deploying FP32 ONNX ensured 0% accuracy loss while maintaining a sub-200 ms latency.

---

## 🛠️ Hardware Specifications & Bill of Materials (BOM)

The physical device couples a Raspberry Pi 5 with custom fundus optics:

| Component | Function | Cost (INR) |
|---|---|:---:|
| **Raspberry Pi 5 (4GB RAM)** | Primary Edge Compute Engine (Cortex-A76) | ₹7,300 |
| **Official Active Cooler** | Thermal Management during continuous inference | ₹490 |
| **20D Ophthalmic Condensing Lens** | Fundus Image Magnification & Focus | ₹900 |
| **50:50 Beam Splitter Plate (20x20mm)** | Coaxial Light Path Alignment | ₹700 |
| **Dual Illumination Module (IR + White LEDs)** | Target Alignment (IR) & Image Capture (White) | ₹300 |
| **3D-Printed Frame Mount** | Rigid Optical Chassis & Camera Mounting | ₹700 |
| **Total Hardware Cost** | **Complete Portable Diagnostic Unit** | **₹10,390** |

---

## ⏱️ Edge Pipeline Latency Benchmarks (Raspberry Pi 5 / Laptop CPU)

| Pipeline Step | Latency (ms) | Description |
|---|:---:|---|
| **Preprocessing** | 50.0 ms | Black border crop + CLAHE contrast enhancement |
| **Stage 1 GANomaly Gate** | 32.4 ms | FP32 ONNX latent error & residual score computation |
| **Stage 2 VBLL + Grad-CAM** | 75.3 ms | Feature extraction + 30-pass NumPy sampling + CAM heatmap |
| **Total Execution Time** | **~157.7 ms** | **Comfortably under the 200 ms real-time clinical budget** |

---

## 💻 Web Application UI (`demo_app.py`)

The project includes a clinical single-page web app built with Flask and Vanilla CSS/JS:

- **Drag & Drop Upload**: Instant fundus image preview.
- **Dual-Stage Dashboard**: Real-time Gate decision, Severity Grade pill (Grades 0–4 color-coded), Bayesian posterior distribution bar chart, and timing breakdown.
- **Grad-CAM Visualizer**: Side-by-side original fundus image and colorized heatmap overlay with lesion load percentage.
- **Clinical Rationale**: Automatically generates medical evidence descriptions for microaneurysms, hemorrhages, and exudates.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- `pip install -r deploy/raspberry_pi/requirements_pi.txt`

### 2. Running the Inference CLI
```bash
cd deploy/raspberry_pi
python inference.py "path/to/fundus_image.jpg" --models models --force
```

### 3. Launching the Web Application
```bash
cd deploy/raspberry_pi
python demo_app.py --models models --port 5000
```
Open **`http://localhost:5000`** in your browser to view the clinical dashboard.

---

## 📂 Repository Structure

```
DR_KAGGLE/
├── deploy/
│   └── raspberry_pi/
│       ├── demo_app.py          # Flask clinical web app & UI
│       ├── inference.py         # 2-Stage pipeline & Grad-CAM engine
│       ├── preprocess.py        # Fundus crop & CLAHE preprocessing
│       ├── requirements_pi.txt  # Lightweight dependency list
│       └── models/
│           ├── ganomaly_fp32.onnx
│           ├── stage1_deploy.json
│           ├── stage2_vbll_fp32.onnx
│           ├── vbll_head.npz
│           └── stage2_deploy.json
├── notebooks/
│   ├── stage2_vbll_efficientnetb0.ipynb
│   ├── export_stage1_ganomaly.ipynb
│   └── export_stage2_vbll.ipynb
└── README.md
```

---

## 📜 License & Citation

Distributed under the MIT License. Developed for research and clinical decision support in low-resource healthcare settings.
