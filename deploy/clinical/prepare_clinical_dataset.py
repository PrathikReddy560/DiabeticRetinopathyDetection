"""Convert a hospital dataset export into the training pipeline's index schema.

Outputs a CSV with columns: path, grade, source, split
(ready to concatenate with the project dataframe in the training notebooks).

Examples:
    python prepare_clinical_dataset.py --root clinic_data --layout csv \
        --csv labels.csv --images images --out clinical_index.csv
    python prepare_clinical_dataset.py --root clinic_data --layout folders \
        --out clinical_index.csv
"""
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def from_csv(root, csv_name, images_name):
    csv_path = root / csv_name
    images_dir = root / images_name
    if not csv_path.exists():
        sys.exit(f"ERROR: CSV not found: {csv_path}")
    if not images_dir.exists():
        sys.exit(f"ERROR: image directory not found: {images_dir}")

    df = pd.read_csv(csv_path)
    required = {"id_code", "diagnosis"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV missing columns {missing}; found {list(df.columns)}")

    rows, unmatched = [], 0
    for r in df.itertuples(index=False):
        image_id = str(r.id_code)
        grade = int(r.diagnosis)
        candidates = [images_dir / f"{image_id}{e}" for e in
                      (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            unmatched += 1
            continue
        rows.append({"path": str(path), "grade": grade, "source": "clinical", "split": "all"})

    if unmatched:
        print(f"WARNING: {unmatched} CSV rows had no matching image (skipped)")
    return pd.DataFrame(rows)


def from_folders(root):
    rows = []
    for grade in range(5):
        gdir = root / str(grade)
        if not gdir.exists():
            print(f"WARNING: missing grade folder {gdir} (skipped)")
            continue
        for p in gdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rows.append({"path": str(p), "grade": grade, "source": "clinical", "split": "all"})
    return pd.DataFrame(rows)


def validate(df):
    if df.empty:
        sys.exit("ERROR: no images indexed - check the layout and paths")

    dupes = df.duplicated(subset="path").sum()
    if dupes:
        df = df.drop_duplicates(subset="path").reset_index(drop=True)
        print(f"NOTE: dropped {dupes} duplicate path entries")

    bad_grades = sorted(set(df["grade"]) - set(range(5)))
    if bad_grades:
        sys.exit(f"ERROR: invalid grade values {bad_grades} (must be 0-4)")

    unreadable = 0
    sample = df.sample(min(200, len(df)), random_state=0)
    for p in sample["path"]:
        if cv2.imread(str(p)) is None:
            unreadable += 1
    if unreadable:
        print(f"WARNING: {unreadable}/{len(sample)} sampled images unreadable by OpenCV "
              "(corrupt or unsupported format)")

    counts = df["grade"].value_counts().sort_index()
    print("\nGrade distribution:")
    for g, n in counts.items():
        print(f"  grade {g}: {n} images ({n / len(df):.1%})")
    if counts.max() / max(counts.min(), 1) > 20:
        print("NOTE: severe class imbalance - the training notebooks handle this with "
              "WeightedRandomSampler, but more grade 3/4 data would help")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root folder")
    ap.add_argument("--layout", choices=["csv", "folders"], required=True)
    ap.add_argument("--csv", default="labels.csv", help="(csv layout) labels CSV filename")
    ap.add_argument("--images", default="images", help="(csv layout) image folder name")
    ap.add_argument("--out", default="clinical_index.csv")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"ERROR: root not found: {root}")

    df = from_csv(root, args.csv, args.images) if args.layout == "csv" else from_folders(root)
    df = validate(df)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}: {len(df)} images indexed "
          f"(columns: path, grade, source, split)")


if __name__ == "__main__":
    main()
