"""Benchmark the pipeline on real hardware: per-stage and combined latency.

Usage:
    python benchmark.py --models models --images /path/to/test/images --limit 50
    python benchmark.py --models models --synthetic 30      # random images, timing only
"""
import argparse
import time
from pathlib import Path

import numpy as np

from inference import DRPipeline

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def percentile(values, p):
    return float(np.percentile(np.array(values), p)) if values else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--images", default=None, help="folder of test images")
    ap.add_argument("--synthetic", type=int, default=0, help="use N random synthetic images")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    pipe = DRPipeline(args.models, threads=args.threads)

    paths = []
    tmp_dir = None
    if args.images:
        paths = [p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXTS][:args.limit]
    elif args.synthetic:
        import cv2
        tmp_dir = Path("_bench_tmp")
        tmp_dir.mkdir(exist_ok=True)
        rng = np.random.default_rng(0)
        for i in range(args.synthetic):
            img = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
            cv2.imwrite(str(tmp_dir / f"synth_{i}.png"), img)
        paths = sorted(tmp_dir.glob("*.png"))[:args.limit]
    if not paths:
        raise SystemExit("No images found - pass --images or --synthetic")

    rows = []
    for p in paths:
        rows.append(pipe.run(p))

    stages = ["preprocess_ms", "stage1_ms", "stage2_ms", "total_ms"]
    print(f"\nBenchmark on {len(rows)} images (threads={args.threads})")
    print(f"{'stage':<16}{'mean':>10}{'p50':>10}{'p95':>10}")
    for s in stages:
        vals = [r["timings"][s] for r in rows if s in r["timings"]]
        if vals:
            print(f"{s:<16}{np.mean(vals):>9.1f}{percentile(vals, 50):>9.1f}{percentile(vals, 95):>9.1f}")

    flagged = [r for r in rows if "severity" in r]
    print(f"\nGate: {len(flagged)}/{len(rows)} images forwarded to Stage 2")
    print(f"Combined worst-case (flagged images): "
          f"mean {np.mean([r['timings']['total_ms'] for r in flagged]) if flagged else 0:.1f} ms, "
          f"p95 {percentile([r['timings']['total_ms'] for r in flagged], 95) if flagged else 0:.1f} ms")

    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
