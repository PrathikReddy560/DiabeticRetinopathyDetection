"""Flask web demo for the DR screening pipeline.

Run on the Raspberry Pi (or on the laptop for rehearsal):
    python demo_app.py --models models --host 0.0.0.0 --port 5000

Then open http://<pi-ip>:5000 from your laptop browser on the same network.
"""
import argparse
import base64
import tempfile
from pathlib import Path

from flask import Flask, request

from inference import DRPipeline

PAGE = """<!doctype html>
<html><head><title>DR Screening - Raspberry Pi</title>
<style>
 body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; background: #f7f7f9; }
 .card { background: #fff; border-radius: 12px; padding: 1.5em; margin-bottom: 1em; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
 h1 { font-size: 1.4em; } h2 { font-size: 1.1em; margin-top: 0; }
 .banner { padding: .8em 1em; border-radius: 8px; font-weight: 600; font-size: 1.1em; }
 .ok { background: #e2f5e5; color: #1c6b28; }
 .warn { background: #fdeaea; color: #a02020; }
 .mid { background: #fff4d6; color: #8a6100; }
 img { max-width: 320px; border-radius: 8px; }
 table { border-collapse: collapse; } td, th { padding: 4px 12px; text-align: left; }
 .bar { background: #e5e7eb; border-radius: 4px; height: 14px; }
 .bar > div { background: #2563eb; height: 14px; border-radius: 4px; }
 code { background: #eee; padding: 1px 5px; border-radius: 4px; }
</style></head><body>
<h1>Diabetic Retinopathy Screening - Edge Demo (INT8, CPU-only)</h1>
<div class="card">
 <form method="post" action="/analyze" enctype="multipart/form-data">
  <input type="file" name="image" accept="image/*" required>
  <button type="submit">Analyze</button>
 </form>
 <p><small>Stage 1 GANomaly gate -> Stage 2 EfficientNet-B0 + VBLL severity with uncertainty.</small></p>
</div>
{{ result }}
</body></html>"""


def render_result(res, img_b64):
    gate_ok = res["gate"] == "normal_gate"
    html = [
        '<div class="card">',
        f'<img src="data:image/jpeg;base64,{img_b64}" alt="fundus">',
        "<h2>Stage 1 - Anomaly Gate</h2>",
        f'<table><tr><td>Anomaly score</td><td><b>{res["anomaly_score"]:.4f}</b></td></tr>',
        f'<tr><td>Threshold</td><td>{res["threshold"]:.4f}</td></tr>',
        f'<tr><td>Gate decision</td><td><b>{"NORMAL" if gate_ok else "FLAGGED -> Stage 2"}</b></td></tr></table>',
    ]
    if "severity" in res:
        s = res["severity"]
        action_warn = res["action"].startswith("Refer")
        html += [
            "<h2>Stage 2 - Severity Grading (VBLL)</h2>",
            f'<div class="banner {"warn" if action_warn else "mid"}">{res["action"]}</div>',
            "<table>",
            f'<tr><td>Predicted grade</td><td><b>{s["grade"]} - {s["grade_name"]}</b></td></tr>',
            f'<tr><td>Confidence</td><td>{s["confidence_pct"]}% (posterior std {s["top_std"]})</td></tr>',
            "</table>",
            "<h2>Grade probabilities</h2><table>",
        ]
        for name, p in s["probabilities"].items():
            pct = int(round(p * 100))
            html.append(f'<tr><td style="width:140px">{name}</td>'
                        f'<td style="width:300px"><div class="bar"><div style="width:{pct}%"></div></div></td>'
                        f"<td>{p:.3f}</td></tr>")
        html.append("</table>")
    else:
        html.append(f'<div class="banner ok">{res["action"]}</div>')
    t = res["timings"]
    html += [
        "<h2>Timing</h2><table>",
        "".join(f"<tr><td>{k}</td><td>{v} ms</td></tr>" for k, v in t.items()),
        "</table></div>",
    ]
    return "".join(html)


def create_app(models_dir, threads):
    app = Flask(__name__)
    pipe = DRPipeline(models_dir, threads=threads)

    @app.get("/")
    def index():
        return PAGE.replace("{{ result }}", "")

    @app.post("/analyze")
    def analyze():
        f = request.files.get("image")
        if not f:
            return PAGE.replace("{{ result }}", "<div class='card'>No image uploaded.</div>")
        img_b64 = base64.b64encode(f.read()).decode()
        f.seek(0)
        with tempfile.NamedTemporaryFile(suffix=Path(f.filename).suffix or ".jpg", delete=False) as tmp:
            f.save(tmp.name)
            res = pipe.run(tmp.name)
        return PAGE.replace("{{ result }}", render_result(res, img_b64))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    app = create_app(args.models, args.threads)
    print(f"Demo running on http://{args.host}:{args.port} - open the Pi's IP from your laptop browser")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
