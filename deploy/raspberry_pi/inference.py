"""Two-stage DR screening pipeline for Raspberry Pi (CPU-only, INT8 ONNX).

Stage 1: GANomaly gate (INT8 ONNX, 128x128) -> anomaly score -> threshold gate
Stage 2: EfficientNet-B0 + VBLL (INT8 ONNX, 224x224) -> severity grade 0-4 with
         Bayesian uncertainty sampled in pure NumPy (no PyTorch on device)

Usage:
    python inference.py /path/to/fundus.jpg --models models
    python inference.py /path/to/fundus.jpg --force   # grade even if gate says normal
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from preprocess import preprocess

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class DRPipeline:
    def __init__(self, models_dir="models", threads=4, seed=42):
        models_dir = Path(models_dir)
        with open(models_dir / "stage1_deploy.json") as f:
            self.s1_cfg = json.load(f)
        with open(models_dir / "stage2_deploy.json") as f:
            self.s2_cfg = json.load(f)

        head = np.load(models_dir / "vbll_head.npz")
        self.w_mu = head["w_mu"]                       # (5, 1280)
        self.w_sigma = np.exp(head["w_log_sigma"])     # (5, 1280)
        self.b_mu = head["b_mu"]                       # (5,)
        self.b_sigma = np.exp(head["b_log_sigma"])     # (5,)

        s1_path = models_dir / "ganomaly_fp32.onnx"
        if not s1_path.exists():
            s1_path = models_dir / "ganomaly_int8.onnx"
        s2_path = models_dir / "stage2_vbll_int8.onnx"
        if not s2_path.exists():
            s2_path = models_dir / "stage2_vbll_fp32.onnx"

        self.s1 = _session(s1_path, threads)
        self.s2 = _session(s2_path, threads)
        self.s1_in = self.s1.get_inputs()[0].name
        self.s2_in = self.s2.get_inputs()[0].name

        self.rng = np.random.default_rng(seed)
        self.threshold = float(self.s1_cfg["threshold_youden"])
        self.params = self.s1_cfg["znorm_params"]
        self.grade_names = {int(k): v for k, v in self.s2_cfg["grade_names"].items()}
        self.passes = int(self.s2_cfg.get("predict_passes", 30))
        self.low_conf_std = float(self.s2_cfg.get("low_conf_std", 0.15))

    # ---------------- stage 1: GANomaly gate ----------------
    def stage1_score(self, img128_bgr):
        x = img128_bgr[..., ::-1].astype(np.float32) / 127.5 - 1.0   # BGR->RGB, [-1,1]
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        xhat, z, zhat = self.s1.run(None, {self.s1_in: x})
        latent = float(((z - zhat) ** 2).mean())
        flat = np.abs(x - xhat).mean(axis=1).ravel()
        k = max(1, int(0.01 * flat.size))
        top = float(np.sort(flat)[-k:].mean())
        lz = (latent - self.params["latent"]["median"]) / self.params["latent"]["scale"]
        tz = (top - self.params["top_residual"]["median"]) / self.params["top_residual"]["scale"]
        return 0.5 * lz + 0.5 * tz

    # ---------------- stage 2: severity + VBLL uncertainty ----------------
    def stage2_grade(self, img224_bgr):
        x = img224_bgr[..., ::-1].astype(np.float32) / 255.0         # BGR->RGB
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        _, feats = self.s2.run(None, {self.s2_in: x})
        f = feats[0]                                                 # (1280,)

        # 30 posterior weight samples, pure NumPy
        w = self.w_mu[None] + self.w_sigma[None] * self.rng.standard_normal(
            (self.passes, *self.w_mu.shape)).astype(np.float32)
        b = self.b_mu[None] + self.b_sigma[None] * self.rng.standard_normal(
            (self.passes, self.b_mu.shape[0])).astype(np.float32)
        logits = w @ f + b                                           # (passes, 5)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=1, keepdims=True)

        mean_p = probs.mean(axis=0)
        std_p = probs.std(axis=0)
        grade = int(mean_p.argmax())
        top_std = float(std_p[grade])
        return {
            "grade": grade,
            "grade_name": self.grade_names[grade],
            "probabilities": {self.grade_names[i]: round(float(mean_p[i]), 4) for i in range(5)},
            "confidence_pct": round((1.0 - top_std) * 100.0, 1),
            "top_std": round(top_std, 4),
            "low_confidence": bool(top_std > self.low_conf_std),
        }

    # ---------------- full pipeline ----------------
    def run(self, image_path, force_grade=False):
        timings = {}
        t0 = time.perf_counter()
        img128 = preprocess(image_path, 128)
        img224 = preprocess(image_path, 224)
        timings["preprocess_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        score = self.stage1_score(img128)
        timings["stage1_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        result = {
            "image": str(image_path),
            "anomaly_score": round(score, 4),
            "threshold": self.threshold,
            "gate": "send_stage2" if score > self.threshold else "normal_gate",
        }

        if score > self.threshold or force_grade:
            t0 = time.perf_counter()
            s2 = self.stage2_grade(img224)
            timings["stage2_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            result["severity"] = s2
            result["action"] = ("Refer for Manual Review" if s2["low_confidence"]
                                else f"Proceed - grade {s2['grade']} ({s2['grade_name']})")
        else:
            result["action"] = "NORMAL, stop here"

        timings["total_ms"] = round(sum(timings.values()), 1)
        result["timings"] = timings
        return result


def main():
    ap = argparse.ArgumentParser(description="DR screening pipeline (INT8, CPU-only)")
    ap.add_argument("image", help="path to a fundus image")
    ap.add_argument("--models", default="models", help="directory with the INT8 artifacts")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="grade even if the gate says normal")
    args = ap.parse_args()

    pipe = DRPipeline(args.models, threads=args.threads)
    print(json.dumps(pipe.run(args.image, force_grade=args.force), indent=2))


if __name__ == "__main__":
    main()
