"""Flask web application for the Edge-Deployable DR Screening Pipeline.

Provides a modern clinical web UI with Grad-CAM explainability.

Run locally or on Raspberry Pi 5:
    python demo_app.py --models models --host 0.0.0.0 --port 5000
"""
import argparse
import base64
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

from inference import DRPipeline

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetinaAI — Edge Diabetic Retinopathy Screening</title>
  <meta name="description" content="AI-powered diabetic retinopathy screening with Grad-CAM explainability on Raspberry Pi 5">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 30, 49, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-cyan: #06b6d4;
      --accent-blue: #3b82f6;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: var(--bg);
      background-image: radial-gradient(circle at 20% 10%, rgba(6, 182, 212, 0.08) 0%, transparent 40%),
                        radial-gradient(circle at 80% 80%, rgba(59, 130, 246, 0.06) 0%, transparent 40%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 1.5rem 1rem;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 1.5rem; padding-bottom: 1rem;
      border-bottom: 1px solid var(--card-border);
    }
    .brand { display: flex; align-items: center; gap: 0.75rem; }
    .brand-logo {
      width: 44px; height: 44px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      border-radius: 12px; display: grid; place-items: center;
      font-weight: 700; font-size: 1.5rem; color: #fff;
      box-shadow: 0 0 24px rgba(6, 182, 212, 0.4);
    }
    .brand-title h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }
    .brand-title p { font-size: 0.82rem; color: var(--text-muted); }
    .badge-edge {
      background: rgba(6, 182, 212, 0.12); color: var(--accent-cyan);
      border: 1px solid rgba(6, 182, 212, 0.25); padding: 0.35rem 0.75rem;
      border-radius: 20px; font-size: 0.78rem; font-weight: 600;
    }
    .card {
      background: var(--card-bg); backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--card-border); border-radius: 20px;
      padding: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .card-title {
      font-size: 0.95rem; font-weight: 600; margin-bottom: 1rem;
      display: flex; align-items: center; gap: 0.5rem;
      text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted);
    }
    .top-grid { display: grid; grid-template-columns: 340px 1fr; gap: 1.5rem; }
    @media (max-width: 900px) { .top-grid { grid-template-columns: 1fr; } }

    /* Upload Panel */
    .dropzone {
      border: 2px dashed rgba(255,255,255,0.12); border-radius: 14px;
      padding: 2rem 1rem; text-align: center; cursor: pointer;
      transition: all 0.25s; background: rgba(0,0,0,0.2);
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent-cyan); background: rgba(6, 182, 212, 0.05);
    }
    .dropzone-icon { font-size: 2.5rem; margin-bottom: 0.5rem; opacity: 0.7; }
    .dropzone p { font-size: 0.88rem; color: var(--text-muted); }
    .dropzone span { color: var(--accent-cyan); font-weight: 600; }
    #fileInput { display: none; }
    .btn-submit {
      width: 100%; margin-top: 1.25rem; padding: 0.85rem; border: none;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      color: white; font-family: inherit; font-size: 1rem; font-weight: 600;
      cursor: pointer; transition: opacity 0.2s, transform 0.1s;
      box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3);
    }
    .btn-submit:hover { opacity: 0.92; transform: translateY(-1px); }
    .btn-submit:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
    .preview-container { margin-top: 1rem; display: none; text-align: center; }
    .preview-container img {
      max-width: 100%; max-height: 200px; border-radius: 12px;
      border: 1px solid var(--card-border);
    }

    /* Placeholder */
    .placeholder-state {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; min-height: 320px;
      color: var(--text-muted); text-align: center;
    }
    .placeholder-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.25; }

    /* Action Banner */
    .action-banner {
      padding: 1rem 1.25rem; border-radius: 14px; font-weight: 600;
      font-size: 1.05rem; display: flex; align-items: center;
      justify-content: space-between; margin-bottom: 1.25rem;
    }
    .banner-normal { background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.25); color: #34d399; }
    .banner-proceed { background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.25); color: #60a5fa; }
    .banner-referral { background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.25); color: #fb7185; }

    /* Metrics */
    .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1.25rem; }
    @media (max-width: 700px) { .metrics-grid { grid-template-columns: repeat(2, 1fr); } }
    .metric-card {
      background: rgba(0,0,0,0.25); border: 1px solid var(--card-border);
      border-radius: 12px; padding: 0.85rem;
    }
    .metric-label {
      font-size: 0.7rem; color: var(--text-muted);
      text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem;
    }
    .metric-value { font-size: 1.15rem; font-weight: 700; }
    .grade-badge {
      display: inline-block; padding: 0.25rem 0.65rem;
      border-radius: 8px; font-size: 0.82rem; font-weight: 700;
    }
    .grade-0 { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .grade-1 { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
    .grade-2 { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .grade-3 { background: rgba(249, 115, 22, 0.2); color: #fb923c; }
    .grade-4 { background: rgba(244, 63, 94, 0.2); color: #f87171; }

    /* Explainability Section */
    .explain-section { margin-top: 1.25rem; }
    .explain-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-top: 0.75rem; }
    @media (max-width: 700px) { .explain-grid { grid-template-columns: 1fr; } }
    .explain-card {
      background: rgba(0,0,0,0.2); border: 1px solid var(--card-border);
      border-radius: 14px; overflow: hidden;
    }
    .explain-card img {
      width: 100%; height: auto; display: block;
    }
    .explain-card-label {
      padding: 0.6rem 0.85rem; font-size: 0.8rem; font-weight: 600;
      color: var(--text-muted); display: flex; align-items: center; gap: 0.4rem;
    }

    /* Clinical Explanation */
    .clinical-box {
      background: rgba(6, 182, 212, 0.06); border: 1px solid rgba(6, 182, 212, 0.15);
      border-radius: 14px; padding: 1.15rem; margin-top: 1.25rem;
    }
    .clinical-box h3 {
      font-size: 0.85rem; font-weight: 600; color: var(--accent-cyan);
      margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem;
      text-transform: uppercase; letter-spacing: 0.04em;
    }
    .clinical-box p { font-size: 0.88rem; line-height: 1.6; color: var(--text-muted); }
    .clinical-box strong { color: var(--text-main); }
    .clinical-box .highlight {
      display: inline-block; padding: 0.15rem 0.5rem; border-radius: 6px;
      font-weight: 600; font-size: 0.82rem;
    }

    /* Probability Bars */
    .chart-box { margin-top: 1.25rem; }
    .chart-title { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.75rem; color: var(--text-muted); }
    .bar-row { margin-bottom: 0.6rem; }
    .bar-header { display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 0.25rem; }
    .bar-track { background: rgba(255,255,255,0.05); height: 8px; border-radius: 4px; overflow: hidden; }
    .bar-fill {
      height: 100%; border-radius: 4px;
      transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Lesion Load Meter */
    .lesion-meter { margin-top: 1rem; }
    .lesion-meter-label { font-size: 0.82rem; margin-bottom: 0.3rem; display: flex; justify-content: space-between; }
    .lesion-meter-track {
      background: rgba(255,255,255,0.06); height: 10px; border-radius: 5px;
      overflow: hidden; position: relative;
    }
    .lesion-meter-fill {
      height: 100%; border-radius: 5px;
      transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .lesion-meter-markers {
      position: relative; height: 14px; font-size: 0.65rem; color: var(--text-muted);
    }
    .lesion-marker {
      position: absolute; top: 2px;
      border-left: 1px dashed rgba(255,255,255,0.15); padding-left: 4px;
    }

    /* Timing */
    .timing-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 0.75rem; }
    .timing-table td { padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); color: var(--text-muted); }
    .timing-table tr:last-child td { border: none; font-weight: 700; color: var(--text-main); }
    .timing-table td:last-child { text-align: right; font-variant-numeric: tabular-nums; }

    /* Loader */
    .loader-overlay {
      display: none; position: fixed; inset: 0; background: rgba(11,15,25,0.85);
      z-index: 100; place-items: center;
    }
    .loader-overlay.active { display: grid; }
    .loader-content { text-align: center; }
    .loader-ring {
      width: 56px; height: 56px; margin: 0 auto 1rem;
      border: 3px solid rgba(255,255,255,0.08); border-radius: 50%;
      border-top-color: var(--accent-cyan);
      animation: spin 0.9s ease-in-out infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loader-text { font-size: 0.95rem; color: var(--text-muted); }
    .loader-stage { font-size: 0.82rem; color: var(--accent-cyan); margin-top: 0.5rem; font-weight: 600; }

    /* Fade-in */
    .fade-in { animation: fadeIn 0.5s ease-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <div class="brand-logo">R</div>
        <div class="brand-title">
          <h1>RetinaAI Screening</h1>
          <p>Edge AI Diagnostic Device &mdash; Raspberry Pi 5</p>
        </div>
      </div>
      <div class="badge-edge">&#x1f4a1; XAI-Enabled &middot; Pi 5 (4GB)</div>
    </header>

    <div class="top-grid">
      <div class="card">
        <div class="card-title">&#x1f4f7; Fundus Image Acquisition</div>
        <form id="analyzeForm">
          <div class="dropzone" id="dropzone">
            <div class="dropzone-icon">&#x1f441;&#xfe0f;</div>
            <p>Drag &amp; drop fundus image here</p>
            <p>or <span>browse files</span></p>
            <input type="file" id="fileInput" name="image" accept="image/*" required>
          </div>
          <div class="preview-container" id="previewContainer">
            <img id="previewImg" src="#" alt="Fundus Preview">
          </div>
          <button type="submit" class="btn-submit" id="submitBtn" disabled>
            &#x1f52c; Run Dual-Stage Screening
          </button>
        </form>
      </div>

      <div class="card" id="resultsCard">
        <div class="placeholder-state" id="placeholderState">
          <div class="placeholder-icon">&#x1fa7a;</div>
          <p>Upload a fundus image to run the two-stage<br>AI screening pipeline with Grad-CAM explainability.</p>
        </div>
        <div id="resultsContent" style="display: none;"></div>
      </div>
    </div>

    <div id="explainSection" style="display: none; margin-top: 1.5rem;" class="fade-in"></div>
  </div>

  <div class="loader-overlay" id="loaderOverlay">
    <div class="loader-content">
      <div class="loader-ring"></div>
      <div class="loader-text">Analyzing fundus image&hellip;</div>
      <div class="loader-stage" id="loaderStage">Stage 1: Anomaly Detection Gate</div>
    </div>
  </div>

  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImg = document.getElementById('previewImg');
    const submitBtn = document.getElementById('submitBtn');
    const analyzeForm = document.getElementById('analyzeForm');
    const loaderOverlay = document.getElementById('loaderOverlay');
    const loaderStage = document.getElementById('loaderStage');
    const placeholderState = document.getElementById('placeholderState');
    const resultsContent = document.getElementById('resultsContent');
    const explainSection = document.getElementById('explainSection');

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault(); dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; handleFileSelect(); }
    });
    fileInput.addEventListener('change', handleFileSelect);

    let uploadedImageB64 = null;
    function handleFileSelect() {
      const file = fileInput.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          previewImg.src = e.target.result;
          uploadedImageB64 = e.target.result;
          previewContainer.style.display = 'block';
          submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
      }
    }

    analyzeForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const file = fileInput.files[0];
      if (!file) return;
      const formData = new FormData();
      formData.append('image', file);
      submitBtn.disabled = true;
      loaderOverlay.classList.add('active');
      loaderStage.textContent = 'Stage 1: Anomaly Detection Gate';
      setTimeout(() => { loaderStage.textContent = 'Stage 2: Severity Grading + Grad-CAM'; }, 1200);
      try {
        const response = await fetch('/api/analyze', { method: 'POST', body: formData });
        const res = await response.json();
        renderResults(res);
      } catch (err) {
        alert('Analysis failed: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        loaderOverlay.classList.remove('active');
      }
    });

    const gradeColors = ['#10b981', '#3b82f6', '#f59e0b', '#f97316', '#f43f5e'];
    const gradeDescriptions = {
      0: { short: 'No DR', detail: 'No visible signs of diabetic retinopathy were detected. The retinal vasculature, macula, and optic disc appear within normal limits. No microaneurysms, hemorrhages, or exudates are observed in the regions highlighted by the model.' },
      1: { short: 'Mild NPDR', detail: 'Early signs of non-proliferative diabetic retinopathy detected. The Grad-CAM activation map highlights regions consistent with the presence of <strong>microaneurysms</strong> &mdash; small round dots indicating weakened capillary walls. These are typically the earliest clinical signs of DR.' },
      2: { short: 'Moderate NPDR', detail: 'The model identifies features consistent with moderate non-proliferative diabetic retinopathy. Grad-CAM highlights regions suggesting <strong>dot-blot hemorrhages</strong>, <strong>hard exudates</strong> (lipid deposits), and possible <strong>cotton-wool spots</strong> (retinal nerve fiber layer infarcts). The activated regions indicate vascular leakage patterns.' },
      3: { short: 'Severe NPDR', detail: 'The activation map reveals widespread retinal involvement consistent with severe NPDR. The model detects features matching the <strong>4-2-1 rule</strong>: extensive hemorrhages, venous beading, and intraretinal microvascular abnormalities (IRMA). <strong>Urgent ophthalmology referral recommended</strong>.' },
      4: { short: 'Proliferative DR', detail: 'The Grad-CAM map highlights extensive regions consistent with proliferative diabetic retinopathy. Activated areas suggest <strong>neovascularization</strong> (new abnormal blood vessel growth), possible <strong>vitreous hemorrhage</strong>, and <strong>fibrous proliferation</strong>. <strong>Immediate specialist intervention required to prevent vision loss</strong>.' }
    };

    function getLesionDescription(lesionLoad, grade) {
      const pct = (lesionLoad * 100).toFixed(1);
      if (grade === 0) {
        if (lesionLoad < 0.01) return `Lesion load is <strong>${pct}%</strong> of the retinal area &mdash; minimal activation consistent with a healthy retina. No concerning focal regions detected.`;
        return `Lesion load is <strong>${pct}%</strong>. Minor activation detected but below clinical significance threshold (1%). May reflect image artifacts or normal anatomical variation.`;
      }
      if (lesionLoad < 0.03) return `Lesion load is <strong>${pct}%</strong> of the retinal area. Focal activation in specific regions warrants attention despite low overall coverage.`;
      if (lesionLoad < 0.08) return `Lesion load covers <strong>${pct}%</strong> of the retinal area, showing moderate pathological involvement. Activated regions correlate with vascular abnormalities.`;
      return `Lesion load is <strong>${pct}%</strong> of the retinal area &mdash; significant pathological involvement detected. Widespread activation indicates extensive retinal damage requiring clinical follow-up.`;
    }

    function renderResults(res) {
      placeholderState.style.display = 'none';
      resultsContent.style.display = 'block';
      resultsContent.className = 'fade-in';

      const isNormal = res.gate === 'normal_gate';
      const isReferral = res.action && res.action.startsWith('Refer');
      const hasSeverity = !!res.severity;
      const grade = hasSeverity ? res.severity.grade : -1;

      let bannerClass = isNormal ? 'banner-normal' : (isReferral ? 'banner-referral' : 'banner-proceed');

      let html = `
        <div class="action-banner ${bannerClass}">
          <span>${res.action}</span>
          <span style="font-size:0.82rem;opacity:0.8;">Score: ${res.anomaly_score.toFixed(4)}</span>
        </div>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-label">Stage 1 Gate</div>
            <div class="metric-value">${isNormal ? '&#x2705; Normal' : '&#x26a0;&#xfe0f; Flagged'}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Severity Grade</div>
            <div class="metric-value">
              ${hasSeverity ? '<span class="grade-badge grade-' + grade + '">' + grade + ' &mdash; ' + res.severity.grade_name + '</span>' : 'N/A'}
            </div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">${hasSeverity ? res.severity.confidence_pct + '%' : '100%'}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Total Latency</div>
            <div class="metric-value">${res.timings.total_ms} ms</div>
          </div>
        </div>
      `;

      if (hasSeverity) {
        html += `<div class="chart-box"><div class="chart-title">Bayesian Posterior Probability Distribution (30 Samples)</div>`;
        let idx = 0;
        for (const [name, p] of Object.entries(res.severity.probabilities)) {
          const pct = (p * 100).toFixed(1);
          html += `
            <div class="bar-row">
              <div class="bar-header"><span>${name}</span><span>${pct}%</span></div>
              <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${gradeColors[idx++]};"></div></div>
            </div>`;
        }
        html += `</div>`;
      }

      html += `
        <div style="margin-top:1rem;">
          <div class="chart-title">Edge Pipeline Performance</div>
          <table class="timing-table">
            <tr><td>Preprocessing (CLAHE + Crop)</td><td>${res.timings.preprocess_ms} ms</td></tr>
            <tr><td>Stage 1 GANomaly Gate</td><td>${res.timings.stage1_ms} ms</td></tr>
            ${res.timings.stage2_ms ? '<tr><td>Stage 2 VBLL + Grad-CAM</td><td>' + res.timings.stage2_ms + ' ms</td></tr>' : ''}
            <tr><td>Total Processing</td><td>${res.timings.total_ms} ms</td></tr>
          </table>
        </div>`;

      resultsContent.innerHTML = html;

      // Explainability section
      if (hasSeverity && res.cam_heatmap) {
        const lesionLoad = res.cam_lesion_load || 0;
        const lesionPct = (lesionLoad * 100).toFixed(1);
        const lesionColor = lesionLoad < 0.01 ? 'var(--accent-emerald)' : (lesionLoad < 0.08 ? 'var(--accent-amber)' : 'var(--accent-rose)');

        explainSection.style.display = 'block';
        explainSection.className = 'card fade-in';
        explainSection.innerHTML = `
          <div class="card-title">&#x1f9e0; Explainable AI &mdash; Grad-CAM Activation Map</div>

          <div class="explain-grid">
            <div class="explain-card">
              <div class="explain-card-label">&#x1f441;&#xfe0f; Original Fundus Image</div>
              <img src="${uploadedImageB64}" alt="Original fundus">
            </div>
            <div class="explain-card">
              <div class="explain-card-label">&#x1f525; Grad-CAM Heatmap Overlay</div>
              <img src="data:image/png;base64,${res.cam_heatmap}" alt="Grad-CAM heatmap">
            </div>
          </div>

          <div class="lesion-meter" style="margin-top:1.25rem;">
            <div class="lesion-meter-label">
              <span>Lesion Activation Load</span>
              <span style="font-weight:700;color:${lesionColor};">${lesionPct}%</span>
            </div>
            <div class="lesion-meter-track">
              <div class="lesion-meter-fill" style="width:${Math.min(lesionPct, 100)}%;background:linear-gradient(90deg, var(--accent-emerald), ${lesionColor});"></div>
            </div>
            <div class="lesion-meter-markers">
              <div class="lesion-marker" style="left:1%;">1% (low)</div>
              <div class="lesion-marker" style="left:8%;">8% (high)</div>
              <div class="lesion-marker" style="left:20%;">20%</div>
            </div>
          </div>

          <div class="clinical-box">
            <h3>&#x1f4cb; Clinical Interpretation</h3>
            <p><strong>Predicted: Grade ${grade} &mdash; ${gradeDescriptions[grade].short}</strong></p>
            <p style="margin-top:0.5rem;">${gradeDescriptions[grade].detail}</p>
            <p style="margin-top:0.5rem;">${getLesionDescription(lesionLoad, grade)}</p>
            <p style="margin-top:0.7rem;font-size:0.78rem;color:var(--text-muted);">&#x26a0;&#xfe0f; This AI screening is intended as a clinical decision support tool. All flagged cases should be reviewed by a qualified ophthalmologist. The Grad-CAM visualization shows which retinal regions most influenced the model's prediction.</p>
          </div>
        `;
      } else {
        explainSection.style.display = 'none';
      }
    }
  </script>
</body>
</html>
"""


def create_app(models_dir, threads):
    app = Flask(__name__)
    pipe = DRPipeline(models_dir, threads=threads)

    @app.get("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.post("/api/analyze")
    def analyze():
        f = request.files.get("image")
        if not f:
            return jsonify({"error": "No image uploaded"}), 400

        with tempfile.NamedTemporaryFile(suffix=Path(f.filename).suffix or ".jpg", delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        try:
            res = pipe.run(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return jsonify(res)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


def main():
    ap = argparse.ArgumentParser(description="DR Screening Pipeline Web App")
    ap.add_argument("--models", default="models", help="Directory containing model files")
    ap.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    ap.add_argument("--port", type=int, default=5000, help="Port to listen on")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for ONNX Runtime")
    args = ap.parse_args()

    app = create_app(args.models, args.threads)
    print(f"\n\U0001f680 RetinaAI Web App running at http://{args.host}:{args.port}")
    print(f"\U0001f449 Access from browser: http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
