"""Expected Calibration Error (ECE) evaluation script.

Computes ECE, MCE, Brier Score, and rejection statistics to match
Base Paper 1 (Ramalingam et al.) which reports ECE = 0.0217.

Run on Kaggle (where datasets live) or locally if datasets are present:
    python compute_ece.py --models models --data-root /path/to/datasets

Outputs:
    - ECE, MCE, Brier Score to console
    - Rejection rate at 70% MSP threshold
    - Reliability diagram saved as reliability_diagram.png
"""
import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ─── Paths (edit for your environment) ───────────────────────────────
# These default to Kaggle paths; override with --data-root for local
KAGGLE_IDRID = Path("/kaggle/input/datasets/lakshmiprathik/idrid-516/IDRiD")
KAGGLE_APTOS = Path("/kaggle/input/datasets/mariaherrerot/aptos2019")
KAGGLE_ODIR = Path("/kaggle/input/datasets/lakshmiprathik/odir-5k/ODIR-5K")

# ─── Constants (must match training notebook) ────────────────────────
IMG_SIZE = 224
SEED = 42
ODIR_CAP = 1000
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
GRADE_NAMES = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}
PREDICT_PASSES = 30
NUM_BINS = 15
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ─── Preprocessing (same as training) ────────────────────────────────
def crop_square(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 7
    if mask.sum() > 100:
        ys, xs = np.where(mask)
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), np.uint8)
    y, x = (s - h) // 2, (s - w) // 2
    canvas[y:y + h, x:x + w] = img
    return canvas


def preprocess(path, size=224):
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((size, size, 3), np.uint8)
    img = crop_square(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def to_tensor(img_bgr):
    x = img_bgr[..., ::-1].astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


# ─── Dataset loading (same split logic as training notebook) ─────────
def all_images(root):
    return [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def build_dataset(idrid_root, aptos_root, odir_root):
    """Reproduce the exact dataset + split from the training notebook."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    rows = []

    # IDRiD
    for split in ["train", "validation", "test"]:
        for grade in range(5):
            folder = Path(idrid_root) / split / str(grade)
            if not folder.exists():
                continue
            for p in all_images(folder):
                rows.append({"path": str(p), "grade": grade, "source": "idrid"})

    # APTOS
    aptos_csv = Path(aptos_root) / "train_1.csv"
    aptos_img_dir = Path(aptos_root) / "train_images" / "train_images"
    if not aptos_img_dir.exists():
        aptos_img_dir = Path(aptos_root) / "train_images"
    if aptos_csv.exists():
        df_aptos = pd.read_csv(aptos_csv)
        for row in df_aptos.itertuples(index=False):
            candidates = [aptos_img_dir / f"{row.id_code}.png",
                          aptos_img_dir / f"{row.id_code}.jpg",
                          aptos_img_dir / f"{row.id_code}.jpeg"]
            img_path = next((p for p in candidates if p.exists()), None)
            if img_path:
                rows.append({"path": str(img_path), "grade": int(row.diagnosis), "source": "aptos"})

    # ODIR (capped healthy)
    odir_imgs = all_images(odir_root)
    if len(odir_imgs) > ODIR_CAP:
        idx = np.random.RandomState(SEED).choice(len(odir_imgs), size=ODIR_CAP, replace=False)
        odir_imgs = [odir_imgs[i] for i in idx]
    for p in odir_imgs:
        rows.append({"path": str(p), "grade": 0, "source": "odir"})

    df = pd.DataFrame(rows).drop_duplicates(subset="path").reset_index(drop=True)
    df["grade"] = df["grade"].astype(int)

    # Stratified split: 70/15/15 with seed=42 (same as training)
    train_idx, temp_idx = train_test_split(
        np.arange(len(df)), test_size=0.30, stratify=df["grade"], random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=df.iloc[temp_idx]["grade"], random_state=SEED
    )

    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(f"Total dataset: {len(df)} images")
    print(f"Test split:    {len(test_df)} images")
    print(f"Grade distribution:\n{test_df['grade'].value_counts().sort_index()}")
    return test_df


# ─── ECE computation ─────────────────────────────────────────────────
def compute_ece(confidences, predictions, labels, n_bins=NUM_BINS):
    """Compute Expected Calibration Error with equal-width bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()

        if n_in_bin == 0:
            bin_data.append({"lo": lo, "hi": hi, "n": 0, "acc": 0, "conf": 0, "gap": 0})
            continue

        bin_acc = (predictions[mask] == labels[mask]).mean()
        bin_conf = confidences[mask].mean()
        gap = abs(bin_acc - bin_conf)
        ece += (n_in_bin / len(confidences)) * gap
        mce = max(mce, gap)
        bin_data.append({"lo": lo, "hi": hi, "n": int(n_in_bin), "acc": float(bin_acc),
                         "conf": float(bin_conf), "gap": float(gap)})

    return float(ece), float(mce), bin_data


def compute_brier_score(probs, labels, n_classes=5):
    """Multi-class Brier Score = mean squared error of probability estimates."""
    one_hot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# ─── Reliability diagram ─────────────────────────────────────────────
def plot_reliability_diagram(bin_data, ece, mce, output_path):
    """Save reliability diagram as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping reliability diagram plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Reliability diagram
    bin_confs = [b["conf"] for b in bin_data if b["n"] > 0]
    bin_accs = [b["acc"] for b in bin_data if b["n"] > 0]
    bin_counts = [b["n"] for b in bin_data if b["n"] > 0]

    ax1.bar(bin_confs, bin_accs, width=1.0 / NUM_BINS, alpha=0.6, edgecolor="black",
            color="#06b6d4", label="Model")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax1.set_ylabel("Fraction of Correct Predictions", fontsize=12)
    ax1.set_title(f"Reliability Diagram\nECE = {ece:.4f} | MCE = {mce:.4f}", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Right: Bin histogram
    bin_edges = [(b["lo"] + b["hi"]) / 2 for b in bin_data]
    bin_ns = [b["n"] for b in bin_data]
    ax2.bar(bin_edges, bin_ns, width=1.0 / NUM_BINS, alpha=0.6, edgecolor="black", color="#3b82f6")
    ax2.set_xlabel("Confidence Bin", fontsize=12)
    ax2.set_ylabel("Number of Samples", fontsize=12)
    ax2.set_title("Sample Distribution Across Bins", fontsize=13)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Reliability diagram saved to: {output_path}")


# ─── Main evaluation ─────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ECE Evaluation — Base Paper 1 Benchmark")
    ap.add_argument("--models", default="models", help="Directory with ONNX models + VBLL head")
    ap.add_argument("--data-root", default=None, help="Root directory containing IDRiD/, APTOS/, ODIR-5K/")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", default="reliability_diagram.png")
    args = ap.parse_args()

    # Resolve dataset paths
    if args.data_root:
        idrid = Path(args.data_root) / "IDRiD"
        aptos = Path(args.data_root) / "APTOS"
        odir = Path(args.data_root) / "ODIR-5K"
    else:
        idrid, aptos, odir = KAGGLE_IDRID, KAGGLE_APTOS, KAGGLE_ODIR

    # Load models
    models_dir = Path(args.models)
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3

    s2_path = models_dir / "stage2_vbll_fp32.onnx"
    sess = ort.InferenceSession(str(s2_path), sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    head = np.load(models_dir / "vbll_head.npz")
    w_mu = head["w_mu"]
    w_sigma = np.exp(head["w_log_sigma"])
    b_mu = head["b_mu"]
    b_sigma = np.exp(head["b_log_sigma"])

    rng = np.random.default_rng(SEED)

    # Build test dataset
    test_df = build_dataset(idrid, aptos, odir)

    # Run inference on test set
    all_confs = []
    all_preds = []
    all_labels = []
    all_probs = []
    all_msp = []
    all_entropy = []

    print(f"\nRunning inference on {len(test_df)} test images...")
    for i, row in test_df.iterrows():
        img = preprocess(row["path"], IMG_SIZE)
        x = to_tensor(img)
        _, feats = sess.run(None, {input_name: x})[:2]
        f = feats[0]

        # 30 posterior weight samples
        w = w_mu[None] + w_sigma[None] * rng.standard_normal(
            (PREDICT_PASSES, *w_mu.shape)).astype(np.float32)
        b = b_mu[None] + b_sigma[None] * rng.standard_normal(
            (PREDICT_PASSES, b_mu.shape[0])).astype(np.float32)
        logits = w @ f + b
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=1, keepdims=True)

        mean_p = probs.mean(axis=0)
        pred = int(mean_p.argmax())
        conf = float(mean_p.max())
        eps = 1e-10
        entropy = float(-np.sum(mean_p * np.log(mean_p + eps)))

        all_confs.append(conf)
        all_preds.append(pred)
        all_labels.append(int(row["grade"]))
        all_probs.append(mean_p)
        all_msp.append(conf)
        all_entropy.append(entropy)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_df)}")

    all_confs = np.array(all_confs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_msp = np.array(all_msp)

    # ── Compute metrics ──
    accuracy = float((all_preds == all_labels).mean())
    ece, mce, bin_data = compute_ece(all_confs, all_preds, all_labels)
    brier = compute_brier_score(all_probs, all_labels)

    # Rejection statistics (at 70% MSP threshold, matching Paper 1)
    rejected_mask = all_msp < 0.70
    n_rejected = int(rejected_mask.sum())
    rejection_rate = float(n_rejected / len(all_msp))
    if (~rejected_mask).sum() > 0:
        acc_accepted = float((all_preds[~rejected_mask] == all_labels[~rejected_mask]).mean())
    else:
        acc_accepted = 0.0

    # ── Print results ──
    print("\n" + "=" * 60)
    print("  ECE EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Test samples:              {len(all_labels)}")
    print(f"  Overall Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Expected Calibration Error: {ece:.4f}  (Paper 1 target: 0.0217)")
    print(f"  Maximum Calibration Error:  {mce:.4f}")
    print(f"  Brier Score:               {brier:.4f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Rejection Rate (MSP<0.70): {rejection_rate:.4f} ({rejection_rate*100:.2f}%)")
    print(f"    Paper 1 rejection rate:  25.50%")
    print(f"  Accuracy on Accepted:      {acc_accepted:.4f} ({acc_accepted*100:.2f}%)")
    print(f"    Paper 1 acc on accepted: 89.93%")
    print("=" * 60)

    if ece <= 0.0217:
        print("\n  ✅ ECE BEATS Paper 1 benchmark (0.0217)!")
    else:
        print(f"\n  ⚠️  ECE is {ece:.4f} vs Paper 1's 0.0217 — discuss calibration differences in viva")

    # ── Bin details ──
    print("\nCalibration Bins:")
    print(f"  {'Bin':>12s}  {'N':>6s}  {'Acc':>8s}  {'Conf':>8s}  {'Gap':>8s}")
    for b in bin_data:
        if b["n"] > 0:
            print(f"  ({b['lo']:.2f}, {b['hi']:.2f}]  {b['n']:>6d}  {b['acc']:>8.4f}  {b['conf']:>8.4f}  {b['gap']:>8.4f}")

    # ── Save results ──
    results = {
        "test_samples": len(all_labels),
        "accuracy": round(accuracy, 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "rejection_rate_msp70": round(rejection_rate, 4),
        "accuracy_on_accepted": round(acc_accepted, 4),
        "paper1_ece_target": 0.0217,
        "bins": bin_data,
    }
    results_path = Path(args.output).with_suffix(".json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ── Plot ──
    plot_reliability_diagram(bin_data, ece, mce, args.output)


if __name__ == "__main__":
    main()
